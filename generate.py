import argparse
import torch
from model import StoryAI, ModelConfig
from tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("checkpoints/storyai.pt", map_location=device, weights_only=True)
    tokenizer = BPETokenizer.load("checkpoints/tokenizer.json")
    model = StoryAI(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(ids, max_new_tokens=args.tokens, temperature=args.temperature, top_k=20)
    print(tokenizer.decode(output[0].tolist()))

if __name__ == "__main__":
    main()
