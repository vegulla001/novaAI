"""Self-hosted PyTorch inference endpoint matching the existing application adapter.

One process loads one owned .pt checkpoint. Scale with dedicated GPU workers.
This reference server generates by full-context recomputation and is NOT optimized
for high throughput (no KV cache/batching yet).
"""
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import torch
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from .checkpoint import load_checkpoint
from .tokenizer import ByteTokenizer

checkpoint_path = os.environ.get('NATIVE_CHECKPOINT', 'checkpoints/native_small/model.pt')
api_key = os.environ.get('NATIVE_MODEL_API_KEY', 'local-only-model-secret')
device = os.environ.get('NATIVE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
model = None
model_data = None
executor = ThreadPoolExecutor(max_workers=1)  # serialize GPU access until batching is implemented

@asynccontextmanager
async def lifespan(app):
    global model, model_data
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model, model_data = load_checkpoint(checkpoint_path, device=device)
    yield
    executor.shutdown(wait=True)

app = FastAPI(title='Native Novel LM (PyTorch owned weights)', lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    model: str = 'native-novel-lm'
    messages: list[Message]
    temperature: float = Field(default=.7, gt=0, le=2)
    max_tokens: int = Field(default=300, ge=1, le=8192)
    top_k: int = Field(default=40, ge=0, le=260)

def _infer(req):
    tok = ByteTokenizer()
    # Formatting is part of tokenizer training distribution; align datasets with this format.
    prompt = ''.join(f'<{m.role}>\n{m.content}\n' for m in req.messages) + '<assistant>\n'
    ids = tok.encode(prompt, add_bos=True)[-model.config.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.sample(x, new_tokens=req.max_tokens, temperature=req.temperature, top_k=req.top_k)
    completion = tok.decode(y[0, len(ids):].tolist())
    return {'id':'native-local', 'object':'chat.completion', 'model':'native-novel-lm',
            'choices':[{'index':0,'finish_reason':'length', 'message':{'role':'assistant', 'content':completion}}],
            'usage':{'prompt_tokens':len(ids),'completion_tokens':y.shape[1]-len(ids),
                     'total_tokens':y.shape[1]}}

@app.get('/health')
def health():
    return {'ready':model is not None, 'checkpoint_step':model_data['step'] if model_data else None,
            'device':device}

@app.post('/v1/chat/completions')
async def chat(req:ChatRequest, authorization:str|None=Header(default=None)):
    if api_key and authorization != f'Bearer {api_key}': raise HTTPException(401,'Unauthorized')
    if model is None: raise HTTPException(503,'Model not loaded')
    if not req.messages: raise HTTPException(422,'Messages are required')
    import asyncio
    loop=asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _infer, req)
