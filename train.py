import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from dataset import get_batch, load_tokens
from model import GPT, GPTConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="dataset/processed")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=100)
    args = p.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder = Path(args.data_dir)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    train = load_tokens(folder / "train.npy")
    val = load_tokens(folder / "val.npy")
    if min(len(train), len(val)) <= args.block_size:
        raise ValueError("Not enough tokens: lower --block-size or add novels")
    config = GPTConfig(
        vocab_size=meta["vocab_size"], block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    Path("checkpoints").mkdir(exist_ok=True)
    print(f"Device: {device}; train tokens: {len(train)}; val tokens: {len(val)}")
    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(5):
            x, y = get_batch(val, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate()
            print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f}", flush=True)
            torch.save({"model": model.state_dict(), "config": asdict(config), "step": step},
                       "checkpoints/storyai.pt")

if __name__ == "__main__":
    main()
