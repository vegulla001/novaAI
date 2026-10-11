"""StoryGPT: nanoGPT-compatible decoder-only language model with RoPE, RMSNorm, SwiGLU.
All weights are randomly initialized; no pretrained weights are loaded.
"""
import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 2048
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False  # retained for nanoGPT configuration compatibility
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    swiglu_multiple_of: int = 256


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Normalize in float32 for stability; cast back to input dtype.
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError('RoPE requires an even attention head dimension')
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        return freqs.cos().to(dtype)[None, None, :, :], freqs.sin().to(dtype)[None, None, :, :]


def apply_rope(x, cos, sin):
    # x: (batch, heads, time, head_dim); interleaved even/odd pairs.
    even, odd = x[..., ::2], x[..., 1::2]
    out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError('n_embd must be divisible by n_head')
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        head_dim = config.n_embd // config.n_head
        self.rope = RotaryEmbedding(head_dim, config.rope_theta)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        cos, sin = self.rope(T, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                    dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores.masked_fill(~self.mask[None, None, :T, :T], float('-inf'))
            y = self.attn_dropout(F.softmax(scores, dim=-1)) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Approximate 8/3 * n_embd, rounded to a multiple for efficient matmuls.
        hidden = int(8 * config.n_embd / 3)
        m = config.swiglu_multiple_of
        hidden = m * ((hidden + m - 1) // m)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        if config.block_size < 1 or config.vocab_size < 1:
            raise ValueError('block_size and vocab_size must be positive')
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd, config.rms_norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(('attn.c_proj.weight', 'mlp.w2.weight')):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f'number of parameters: {self.get_num_params() / 1e6:.2f}M')

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters())  # no positional embedding to subtract

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f'Cannot forward sequence of length {T}; block_size={self.config.block_size}')
        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        if not (0 < block_size <= self.config.block_size):
            raise ValueError('new block_size must be positive and no larger than existing block_size')
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'mask'):
                block.attn.mask = block.attn.mask[..., :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError('StoryGPT uses RoPE/RMSNorm/SwiGLU and cannot load GPT-2 weights directly. Use init_from="scratch".')

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{'params': decay, 'weight_decay': weight_decay},
                  {'params': no_decay, 'weight_decay': 0.0}]
        fused = device_type == 'cuda' and 'fused' in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **({'fused': True} if fused else {}))

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # Rough nanoGPT-style A100 BF16 utilization estimate; approximate for SwiGLU.
        N = self.get_num_params()
        c = self.config
        flops_per_token = 6 * N + 12 * c.n_layer * c.n_head * (c.n_embd // c.n_head) * c.block_size
        return flops_per_token * c.block_size * fwdbwd_per_iter / dt / 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        if temperature <= 0:
            raise ValueError('temperature must be > 0')
        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError('top_p must be in (0, 1]')
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                if top_k < 1:
                    raise ValueError('top_k must be >= 1')
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float('-inf'))
            if top_p is not None and top_p < 1:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = (cumulative - F.softmax(sorted_logits, dim=-1)) >= top_p
                sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Friendly names used by this standalone StoryAI project.
ModelConfig = GPTConfig
StoryAI = GPT
