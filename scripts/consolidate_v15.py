#!/usr/bin/env python3
"""Consolidate existing V15 token streams without changing token IDs or weights."""
from pathlib import Path
import hashlib, json, shutil, sys
import numpy as np
BASE=Path('dataset/v15')
OLD=BASE/'tokenizer'
EXPECTED='a9d379df209b1b28194d9ae5314a86fcc52043568402fb39d791290823a34115'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for name in ('tokenizer.json','train.npy','val.npy'):
 dst=BASE/name; src=OLD/name
 if dst.exists() and src.exists() and digest(dst)!=digest(src):
  sys.exit(f'CONFLICT: {dst} differs from {src}; no overwrite performed')
 if not dst.exists():
  if not src.exists():sys.exit(f'MISSING: {src}. Build binaries using python build_v15_data.py --outdir dataset/v15')
  shutil.copy2(src,dst)
report_path=BASE/'report.json'
if not report_path.exists():
 src=OLD/'build_report.json'
 if not src.exists():sys.exit('MISSING: build_report.json; build binaries first')
 shutil.copy2(src,report_path)
report=json.loads(report_path.read_text(encoding='utf-8'))
if digest(BASE/'tokenizer.json')!=EXPECTED:sys.exit('Tokenizer hash does not match previously reported V15 tokenizer. Refusing to mix versions.')
if report['tokenizer_sha256']!=EXPECTED:sys.exit('Report tokenizer hash mismatch')
for split in ('train','val'):
 p=BASE/(split+'.npy'); a=np.load(p,mmap_mode='r',allow_pickle=False)
 if a.ndim!=1 or len(a)!=report['splits'][split]['tokens'] or int(a.max())>=8192:sys.exit(f'Invalid {split} tokens')
 if digest(p)!=report['splits'][split]['sha256']:sys.exit(f'Hash mismatch: {p}')
 report['splits'][split]['file']=str(p)
report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('PASS: consolidated V15 tokenizer, train.npy, val.npy, report.json')
print('Training tokens:',report['splits']['train']['tokens'],'Validation tokens:',report['splits']['val']['tokens'])
print('Old tokenizer folder can be archived after you verify your training config.')
