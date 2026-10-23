import torch
from pathlib import Path
from tokenizers import Tokenizer
from model import GPT, GPTConfig

# ================= CONFIG =================

CHECKPOINT = "checkpoints/sft_100.pt"
TOKENIZER = "dataset/v15/tokenizer.json"
OUTPUT = "storyai_test_chapter.txt"

MAX_NEW_TOKENS = 400
TEMPERATURE = 0.8
TOP_K = 40

# ================= LOAD MODEL =================

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file(TOKENIZER)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

model = GPT(GPTConfig(**checkpoint["model_config"]))
model.load_state_dict(checkpoint["model"])
model = model.to(device).eval()

print(f"Device: {device}")
print(f"Checkpoint step: {checkpoint['step']}")

# ================= STORY PROMPT =================

instruction = (
    "Write a fantasy scene about Prince Arin discovering a hidden "
    "door beneath his castle. Behind the door is a forbidden library "
    "containing evidence that his father erased an ancient kingdom "
    "from history. Include dialogue and a clear ending."
)

prompt = (
    f"Instruction:\n{instruction}\n\n"
    "Response:\n"
)

# ================= TOKENIZE =================

prompt_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
).ids

# Keep generation within the model's context window.
max_prompt_tokens = model.config.block_size - MAX_NEW_TOKENS
prompt_ids = prompt_ids[-max_prompt_tokens:]

x = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device,
)

# ================= GENERATE =================

print("Generating story...")

with torch.inference_mode():
    y = model.generate(
        x,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

generated_ids = y[0].tolist()[len(prompt_ids):]

story = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)

# ================= SAVE =================

Path(OUTPUT).write_text(
    story,
    encoding="utf-8",
)

print("\n========== GENERATED STORY ==========\n")
print(story)
print(f"\nSaved to: {OUTPUT}")