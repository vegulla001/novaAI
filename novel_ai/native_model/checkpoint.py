"""Versioned native checkpoints. State dicts only, no downloaded model artifacts."""
from dataclasses import asdict
from pathlib import Path
import json
import torch
from .model import ModelConfig, NativeNovelLM

def save_checkpoint(path, model, *, step=0, optimizer=None, metadata=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': 'novel-ai-native-v1', 'config': asdict(model.config),
               'model': model.state_dict(), 'step': int(step), 'metadata': metadata or {}}
    if optimizer is not None: payload['optimizer'] = optimizer.state_dict()
    tmp = p.with_suffix(p.suffix+'.tmp')
    torch.save(payload, tmp)
    tmp.replace(p)
    return p

def load_checkpoint(path, *, device='cpu', training=False):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format') != 'novel-ai-native-v1': raise ValueError('Unrecognized checkpoint format')
    model = NativeNovelLM(ModelConfig(**data['config']))
    model.load_state_dict(data['model'], strict=True)
    model.to(device)
    model.train(training)
    return model, data
