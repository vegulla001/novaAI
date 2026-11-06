import re
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from training.v15.model import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/v17_rev2_pilot/sft_best.pt"
DATA = "dataset/v17_rev2/character/train_sft.pt"

torch.manual_seed(1717)

ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = GPT(GPTConfig(**ckpt["model_config"])).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=True)

tok = Tokenizer.from_file(
    "dataset/v15_prose_expanded/tokenizer.json"
)
rows = torch.load(DATA, map_location="cpu", weights_only=True)

# Fixed 16 examples across ownership, location, and knowledge.
examples = []

for row in rows[:16]:
    ids = torch.as_tensor(row["input_ids"], dtype=torch.long)
    labels = torch.as_tensor(row["labels"], dtype=torch.long)
    start = int((labels != -100).nonzero()[0].item())

    prompt = ids[:start].tolist()
    expected_ids = labels[start:].tolist()
    expected = tok.decode(
        [v for v in expected_ids if v != 2],
        skip_special_tokens=True,
    )

    examples.append((ids, labels, prompt, expected))

def make_batch(items):
    max_len = max(len(e[0]) - 1 for e in items)
    x = torch.zeros(len(items), max_len, dtype=torch.long)
    y = torch.full((len(items), max_len), -1, dtype=torch.long)

    for i, (ids, labels, _, _) in enumerate(items):
        n = len(ids) - 1
        x[i, :n] = ids[:-1]
        y[i, :n] = torch.where(
            labels[1:] == -100,
            -1,
            labels[1:],
        )

    return x.to(DEVICE), y.to(DEVICE)

def normalize(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))

@torch.inference_mode()
def evaluate(step):
    model.eval()
    correct = 0
    wrong = []

    for _, _, prompt, expected in examples:
        index = torch.tensor(
            [prompt], device=DEVICE, dtype=torch.long
        )

        output = model.generate(
            index,
            max_new_tokens=64,
            temperature=1.0,
            top_k=1,
            eos_id=2,
        )

        generated_ids = output[0, len(prompt):].tolist()
        if 2 in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(2)]

        generated = tok.decode(
            generated_ids, skip_special_tokens=True
        )

        if normalize(generated) == normalize(expected):
            correct += 1
        else:
            wrong.append((expected, generated))

    print(f"\nSTEP {step}: EXACT ANSWERS {correct}/16", flush=True)
    for expected, generated in wrong[:3]:
        print("EXPECTED :", repr(expected), flush=True)
        print("GENERATED:", repr(generated), flush=True)

    model.train()
    return correct

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.0,
)

evaluate(0)

for step in range(1, 151):
    optimizer.zero_grad(set_to_none=True)

    # Every optimizer step sees all 16 fixed examples.
    # Use small microbatches to limit GPU memory usage.
    total_loss = 0.0
    for offset in range(0, 16, 2):
        x, y = make_batch(examples[offset:offset + 2])
        _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        (loss / 8).backward()
        total_loss += float(loss.item()) / 8

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )
    optimizer.step()

    if step in (25, 50, 100, 150):
        print("TRAIN LOSS:", round(total_loss, 4), flush=True)
        correct = evaluate(step)
        if correct == 16:
            print("\nCHARACTER LEARNING TEST: PASS")
            break
else:
    print("\nCHARACTER LEARNING TEST: NOT YET PASSED")

print("Original checkpoint was not modified.")
