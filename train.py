import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from dataset import load_data, get_batch
from model import ModelConfig, StoryAI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/stories.txt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, tokenizer = load_data(args.data)
    tokenizer.save("checkpoints/tokenizer.json")

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = StoryAI(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def evaluate():
        model.eval()
        scores = []
        for _ in range(10):
            x, y = get_batch(val_data, args.batch_size, cfg.block_size, device)
            _, loss = model(x, y)
            scores.append(loss.item())
        model.train()
        return sum(scores) / len(scores)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            print(f"step {step:5d} | train loss {loss.item():.4f} | val loss {evaluate():.4f}")
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg), "step": step},
                "checkpoints/storyai.pt",
            )

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    main()
