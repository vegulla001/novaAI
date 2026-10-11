import argparse
import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The ")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("checkpoints/storyai.pt", map_location=device, weights_only=True)
    tokenizer = BPETokenizer.load("dataset/processed/tokenizer.json")
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=30)
    print(tokenizer.decode(output[0].tolist()))

if __name__ == "__main__":
    main()
