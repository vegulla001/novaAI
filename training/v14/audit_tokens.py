"""Audit corpus, tokenizer, and dataset for control marker contamination."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from tokenizer import BPETokenizer

MARKER = re.compile(r'<(?:GENRE[_A-Z]*|INSTRUCTION|STORY|END_STORY)[^>]*>')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    p.add_argument('--data-dir', default='dataset/processed')
    p.add_argument('--samples', type=int, default=5)
    args = p.parse_args()
    tok = BPETokenizer.load(args.tokenizer)
    print('Tokenizer SHA256:', hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest())
    print('Vocabulary size:', tok.vocab_size)
    for name in ('train', 'val'):
        path = Path(args.data_dir) / (name + '.npy')
        data = np.load(path, mmap_mode='r')
        print(f'\n{name}: {len(data):,} tokens; token range {int(data.min())}..{int(data.max())}')
        if data.min() < 0 or data.max() >= tok.vocab_size:
            raise ValueError(f'{name} contains invalid token IDs')
        # Sample deterministic positions to avoid loading large arrays in memory.
        length = min(1200, len(data))
        starts = np.linspace(0, len(data) - length, max(1, args.samples), dtype=int)
        for i, start in enumerate(starts):
            decoded = tok.decode(data[start:start+length])
            markers = MARKER.findall(decoded)
            bad_utf8 = decoded.count('\ufffd')
            print(f'  sample {i+1}: markers={markers[:10]}, replacement_chars={bad_utf8}')
            print('  text:', repr(decoded[:280]))
    print('\nNOTE: replacement characters at the edges of arbitrary byte-token samples may be normal.')


if __name__ == '__main__':
    main()
