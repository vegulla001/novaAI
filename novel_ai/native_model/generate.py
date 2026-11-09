"""CLI generation from owned local model weights."""
import argparse
import torch
from .checkpoint import load_checkpoint
from .tokenizer import ByteTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/native_small/model.pt')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model, payload = load_checkpoint(args.checkpoint, device=device)
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(args.prompt, add_bos=True)[-model.config.context_length:]
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    output = model.sample(prompt, new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(output[0, len(ids):].tolist()))
    print('\n[checkpoint step:',payload['step'],']')
if __name__ == '__main__': main()
