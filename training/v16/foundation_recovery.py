"""Guarded preparation and opt-in execution of the V16 foundation recovery run.

Place under training/v16/. Never changes an existing checkpoint unless --execute is passed.
"""
import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_summary(path, vocab):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.ndim != 1 or a.dtype.kind not in 'iu':
        raise ValueError(f'{path} must be a one-dimensional integer-token array')
    if len(a) < 1026:
        raise ValueError(f'{path} has insufficient tokens')
    # Checking all token IDs is intentional; prevents CUDA embedding crashes.
    lo, hi = int(a.min()), int(a.max())
    if lo < 0 or hi >= vocab:
        raise ValueError(f'{path}: out-of-range token IDs {lo}..{hi} for vocab={vocab}')
    return a, {'count': int(len(a)), 'dtype': str(a.dtype), 'min': lo, 'max': hi, 'sha256': digest(path)}


def sampled_overlap(train, val, seed=3107, window=256, samples=1200):
    """Partial collision scan only, never a proof of dataset disjointness."""
    rng = np.random.default_rng(seed)
    def fingerprints(arr):
        count = min(samples, max(1, len(arr) - window))
        starts = rng.integers(0, len(arr) - window, size=count)
        return [hashlib.sha256(np.asarray(arr[int(i):int(i)+window]).tobytes()).hexdigest() for i in starts]
    train_fp = fingerprints(train)
    val_fp = fingerprints(val)
    return {
        'sample_window_tokens': window,
        'train_samples': len(train_fp), 'val_samples': len(val_fp),
        'train_unique_sample_rate': round(len(set(train_fp))/len(train_fp), 4),
        'val_unique_sample_rate': round(len(set(val_fp))/len(val_fp), 4),
        'sampled_train_val_collisions': len(set(train_fp) & set(val_fp)),
        'warning': 'Sampled identical windows only; cannot rule out leakage, near duplicates or overlapping books.'
    }


def inspect(checkpoint, data_dir):
    root = Path(data_dir)
    report_path, tok_path = root / 'build_report.json', root / 'tokenizer.json'
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    c = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(c, dict) or not isinstance(c.get('model'), dict):
        raise ValueError('Checkpoint must be a dictionary containing model weights')
    if c.get('mode') != 'pretrain':
        raise ValueError('Foundation checkpoint must have mode=pretrain')
    if int(c.get('step', 0)) <= 0:
        raise ValueError('Refusing pretraining checkpoint with step <= 0')
    cfg = c['model_config']
    token_hash = digest(tok_path)
    if c.get('tokenizer_sha256') != token_hash or report.get('tokenizer_sha256') != token_hash:
        raise ValueError('Checkpoint, build report and tokenizer SHA256 do not all match')
    vocab = int(cfg['vocab_size'])
    if int(report.get('vocab_size', -1)) != vocab:
        raise ValueError('Dataset report vocabulary disagrees with checkpoint')
    train, train_info = array_summary(root / 'train.npy', vocab)
    val, val_info = array_summary(root / 'val.npy', vocab)
    if int(cfg['block_size']) < 1:
        raise ValueError('Invalid model block size')
    if train_info['count'] <= cfg['block_size'] or val_info['count'] <= cfg['block_size']:
        raise ValueError('Not enough tokens for model context length')
    overlaps = sampled_overlap(train, val)
    return {'checkpoint': str(checkpoint), 'checkpoint_step': int(c['step']),
            'config': cfg, 'tokenizer_sha256': token_hash, 'train': train_info,
            'val': val_info, 'overlap_screen': overlaps,
            'checkpoint_has_optimizer': isinstance(c.get('optimizer'), dict),
            'selection': 'weights-only restart: intentionally fresh optimizer and learning-rate schedule',
            'limitations': ['No semantic/copyright check of corpus was performed',
                            'A fixed validation loss and coherent generation must improve before further SFT']}


def command(a, cfg):
    # The stock trainer correctly supports restart-from, but counts NEW steps from zero.
    return [sys.executable, '-u', '-m', 'training.v15.train',
            '--mode', 'pretrain', '--data-dir', a.data_dir,
            '--restart-from', a.checkpoint, '--out-dir', a.out_dir,
            '--block-size', str(cfg['block_size']),
            '--batch-size', str(a.batch_size), '--grad-accum', str(a.grad_accum),
            '--max-steps', str(a.steps), '--lr-decay-steps', str(a.steps),
            '--lr', str(a.lr), '--min-lr', str(a.min_lr),
            '--warmup-steps', str(a.warmup), '--eval-every', str(a.eval_every),
            '--eval-batches', str(a.eval_batches), '--save-every', str(a.save_every),
            '--device', a.device]


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/pretrain_2000.pt')
    p.add_argument('--data-dir', default='dataset/v15_prose_expanded')
    p.add_argument('--out-dir', default='checkpoints/v16_foundation_recovery_01')
    p.add_argument('--report', default='reports/v16_foundation_recovery_preflight.json')
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=8e-5)
    p.add_argument('--min-lr', type=float, default=8e-6)
    p.add_argument('--warmup', type=int, default=40)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--eval-batches', type=int, default=20)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--execute', action='store_true', help='Explicitly begin paid or CPU training')
    return p


def main(argv=None):
    a = make_parser().parse_args(argv)
    if min(a.steps, a.batch_size, a.grad_accum, a.eval_every, a.save_every, a.eval_batches) < 1:
        raise ValueError('All training counts must be positive')
    if not (0 < a.min_lr <= a.lr) or a.warmup >= a.steps or a.warmup < 0:
        raise ValueError('Require 0 < min_lr <= lr and 0 <= warmup < steps')
    out = Path(a.out_dir)
    if out.exists() and any(out.glob('*.pt')):
        raise FileExistsError(f'Refusing to write into existing checkpoint directory: {out}')
    info = inspect(a.checkpoint, a.data_dir)
    cmd = command(a, info['config'])
    info['training_plan'] = {'new_steps': a.steps, 'effective_tokens_per_step':
        a.batch_size * a.grad_accum * int(info['config']['block_size']),
        'planned_token_positions': a.steps * a.batch_size * a.grad_accum * int(info['config']['block_size']),
        'corpus_equivalents': round(a.steps * a.batch_size * a.grad_accum * int(info['config']['block_size']) / info['train']['count'], 2),
        'lr': a.lr, 'min_lr': a.min_lr, 'warmup': a.warmup,
        'baseline_2026_11_05_reference': {'train': 3.7498650789260863, 'val': 3.3318411111831665,
                                         'caution': 'Prior eval used batch-size 2 and different fixed windows; compare against the new BASELINE_FIXED.'}}
    info['command'] = cmd
    target = Path(a.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('PREFLIGHT PASSED: source checkpoint, tokenizer, vocab and token arrays compatible')
    print('TRAIN TOKEN POSITIONS:', info['training_plan']['planned_token_positions'])
    print('SAMPLED TRAIN/VAL COLLISIONS:', info['overlap_screen']['sampled_train_val_collisions'])
    print('COMMAND:', subprocess.list2cmdline(cmd) if sys.platform == 'win32' else shlex.join(cmd))
    print('REPORT:', target)
    if not a.execute:
        print('NO TRAINING RUN. Add --execute on the GPU only when ready.')
        return
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; aborting without running training')
    print('STARTING FOUNDATION RECOVERY; checkpoint source remains unchanged', flush=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
