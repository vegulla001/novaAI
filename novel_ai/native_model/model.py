"""Native PyTorch decoder-only Transformer. No pretrained/model-library weights.

Features: pre-norm RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings,
PyTorch SDPA, gradient checkpointing, optional loss. No KV cache (prefill-only decode).
"""
from dataclasses import dataclass, asdict
from typing import Optional
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    context_length: int = 8192
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    n_layers: int = 32
    intermediate_size: int = 14336
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError('d_model must divide by n_heads and heads by kv_heads')
        if (self.d_model // self.n_heads) % 2: raise ValueError('head dimension must be even')
        if min(self.vocab_size, self.context_length, self.n_layers, self.intermediate_size, self.n_kv_heads) < 1:
            raise ValueError('invalid nonpositive dimension')

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight

def rotary(x, positions, theta):
    # x: [batch, heads, sequence, head_dim]
    d = x.shape[-1]
    freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    angles = torch.outer(positions.float(), freq)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    xf = x.float()
    a, b = xf[..., ::2], xf[..., 1::2]
    output = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    return output.to(dtype=x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.n_kv_heads = c.n_heads, c.n_kv_heads
        self.head_dim = c.d_model // c.n_heads
        self.q_proj = nn.Linear(c.d_model, c.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.dropout, self.theta = c.dropout, c.rope_theta

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        q, k = rotary(q, pos, self.theta), rotary(k, pos, self.theta)
        # Explicit repeat allows CPU tests; production GPU deployments should benchmark
        # enable_gqa=True to reduce K/V duplication where backend supports it.
        if self.n_heads != self.n_kv_heads:
            nrep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(nrep, dim=1)
            v = v.repeat_interleave(nrep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
             dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, -1))

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.intermediate_size, bias=False)
        self.up = nn.Linear(c.d_model, c.intermediate_size, bias=False)
        self.down = nn.Linear(c.intermediate_size, c.d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model, c.rms_eps)
        self.attn = GroupedQueryAttention(c)
        self.ffn_norm = RMSNorm(c.d_model, c.rms_eps)
        self.ffn = SwiGLU(c)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

class NativeNovelLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or idx.shape[1] == 0 or idx.shape[1] > self.config.context_length:
            raise ValueError('Input must be [batch, 1..context_length]')
        x = self.embed(idx)
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.inference_mode()
    def sample(self, prefix, new_tokens=128, temperature=0.8, top_k=40, eos_id=None):
        if temperature <= 0 or top_k < 0: raise ValueError('Invalid sampling settings')
        if new_tokens < 0: raise ValueError('new_tokens cannot be negative')
        self.eval()
        for _ in range(new_tokens):
            logits, _ = self(prefix[:, -self.config.context_length:])
            scores = logits[:, -1].float() / temperature
            if top_k:
                vals, _ = torch.topk(scores, min(top_k, scores.size(-1)))
                scores = scores.masked_fill(scores < vals[:, -1:], -torch.inf)
            nxt = torch.multinomial(F.softmax(scores, dim=-1), num_samples=1)
            prefix = torch.cat((prefix, nxt), dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()): break
        return prefix

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

PRESETS = {
    'smoke': dict(vocab_size=260, context_length=128, d_model=128,
                  n_heads=4, n_kv_heads=2, n_layers=2, intermediate_size=384),
    '1b': dict(d_model=2048, n_heads=16, n_kv_heads=4, n_layers=20, intermediate_size=5632),
    '8b': dict(d_model=4096, n_heads=32, n_kv_heads=8, n_layers=32, intermediate_size=14336),
    '32b': dict(d_model=5120, n_heads=40, n_kv_heads=8, n_layers=64, intermediate_size=27648),
    '70b': dict(d_model=8192, n_heads=64, n_kv_heads=8, n_layers=80, intermediate_size=28672),
}

def config_for(size, **overrides):
    if size not in PRESETS: raise ValueError(f'Unknown size: {size}')
    return ModelConfig(**(dict(PRESETS[size]) | overrides))

def estimate_params(c):
    hd = c.d_model // c.n_heads
    attention = c.d_model * ((c.n_heads + 2*c.n_kv_heads)*hd + c.d_model)
    ffn = 3*c.d_model*c.intermediate_size
    norms = 2*c.d_model
    return c.vocab_size*c.d_model + c.n_layers*(attention + ffn + norms) + c.d_model
