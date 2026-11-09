# Novel AI Platform (PyTorch-backed, production-oriented reference implementation)

**What this is:** a complete Python application/reference codebase for generating a novel from a user prompt. Features include REST API, async Celery jobs, PostgreSQL checkpoints, story planning, multi-chapter continuation, character/timeline memory, automated critique/revision, OIDC authentication and tenant scoping, Markdown export, model-server integration, LoRA SFT training and basic evaluation.

**What this is not:** an audited production deployment, a newly pretrained frontier model, or a guaranteed long-novel consistency solution. Production requirements still include schema migrations, TLS/ingress, stronger rate limits and quotas, backups, tracing, load tests, tenant isolation audits, deployments/monitoring, content policy controls, billing, and formal human narrative evaluation.

## Architecture

```
Browser / client -> FastAPI (JWT + tenant checks) -> Postgres novel, chapters, story state
                                         |            ^
                                         v            |
                                      Redis -> Celery worker -> Native PyTorch model HTTP -> PyTorch GPU
                                                    |      ^
                                                    +---- checkpoints
```

The state machine is `queued -> running -> completed`, with `failed`, `cancelled`, and `resume` support. Each job runs one generation stage (plan or chapter), persists a checkpoint and schedules the next task. A recovery reconciler can republish lost tasks. A per-novel DB lease limits concurrent work. Each written chapter has a uniqueness constraint for idempotency. **Important:** leases are time-based; configure `WORKER_LEASE_SECONDS` to exceed the maximum combined inference duration, or add lease heartbeats for production. The system is an engineered starting point, not exactly-once distributed execution.

## Local setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[test]"
Copy-Item .env.example .env
# Edit .env: set DEV_API_TOKEN to a long random secret, DB URL etc.
```

Start Postgres and Redis, or use Docker Compose. Run development schema initialization:

```powershell
python -m novel_ai.init_db
uvicorn novel_ai.api:app --reload --port 8080
# in a second terminal:
celery -A novel_ai.worker.celery_app worker -Q novel --loglevel INFO --pool=solo
# in a third terminal: python -m novel_ai.reconcile (periodically)
```

For Linux production workers use a supported Celery prefork pool and set concurrency based on model throughput. Do not run a GPU model in each Celery process; have them use a dedicated inference server.

### Legacy third-party model option (not required; skip for an owned model)

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8001 --api-key local-only-model-secret
```

Set `MODEL_BASE_URL=http://localhost:8001/v1`, `MODEL_NAME`, and `MODEL_API_KEY` in `.env`. Choose a model whose license allows your usage. For better novel quality use an appropriately evaluated larger instruct model with sufficient context and GPU resources; **the 1.5B default is for smoke tests, not commercially competitive fiction**. Do not expose vLLM directly to the internet; place it behind private networking and authenticated ingress.

### Quick dev mode without GPUs

The API can be started without a model; jobs will fail until an inference endpoint is present. Unit tests use a model stub so they do not require a GPU. For a full smoke test connect a supported OpenAI-compatible local inference server.

### Start in Docker

1. Place `.env` in the repository root. Set `POSTGRES_PASSWORD` in your shell / secret store. In `.env` configure `MODEL_BASE_URL` to a reachable **private** model-serving address (not `localhost` inside containers).
2. `docker compose -f deploy/compose.yaml up --build -d db redis`
3. `docker compose -f deploy/compose.yaml run --rm api python -m novel_ai.init_db` **development only**
4. `docker compose -f deploy/compose.yaml up --build api worker`
5. Open `http://localhost:8080/docs`; serve `web/index.html` through localhost with a static server and configure its API origin, or visit the `/studio` route from the API.

## API example

```bash
curl -X POST http://localhost:8080/v1/novels \
  -H 'Authorization: Bearer YOUR_DEV_TOKEN' \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: novel-0001' \
  -d '{"prompt":"A city beneath the sea is waking. A cartographer must uncover its history and protect her brother.","genre":"fantasy","tone":"immersive","chapters":5,"words_per_chapter":800}'
# GET /v1/novels/{id}, GET /v1/novels, POST /v1/novels/{id}/cancel,
# POST /v1/novels/{id}/resume, GET /v1/novels/{id}/export
```

## Training on owned/licensed data

- Prepare records with `book_id`, `instruction`, `chapter_outline`, `context`, `continuation`, `license`. Confirm actual copyright/license provenance separately; the script does *not* verify your rights.
- `python training/prepare_dataset.py --input your_records.jsonl --output data/processed`
- `pip install -e '.[training]'`
- `python training/sft_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --train data/processed/train.jsonl --output checkpoints/novel-lora`
- Fine-tuning changes writing style/instruction adherence but **does not by itself guarantee identity consistency**.
- `python training/evaluate.py reviews.jsonl` is a simple name-presence heuristic; use human reviewers and semantic audits for story-quality gates.

## Multi-user security

In development, `DEV_AUTH_ENABLED=true` with a strong `DEV_API_TOKEN` enables a single demo identity. **Never enable this outside trusted local dev.** In production, disable it and configure `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_URL`. Signed RS256 access tokens require `sub` and `tenant_id` claims. All novel read/write operations restrict both tenant and user IDs. Production identity providers may need a claim-mapping adapter. Default browsers must use a real OIDC login; do not collect long-lived JWTs in the sample UI.

## Scaling

- API replicas: stateless FastAPI behind authenticated ingress.
- Worker replicas: scale independently; one chain per novel, queue limits/quotas required.
- Model replicas: separate GPU inference cluster (vLLM) with load balancing, batching, capacity planning.
- Database: managed Postgres, appropriate backups, migrations, retention policies.
- Redis: durable managed configuration and dead-letter monitoring; reconciler recovers scheduling gaps.
- Memory: canonical story bible and per-chapter state; for 100+ chapters add hierarchical summaries, retrieval/vector index and consistency checks.

See `docs/production_checklist.md` for missing enterprise hardening and `docs/evaluation.md` for quality tests.

---

## Native PyTorch model and OWN checkpoints (no third-party foundation model)

**This repository now includes an independently defined, trainable causal Transformer**, an independently implemented UTF-8 byte tokenizer, a checkpoint loader/saver, a local HTTP inference server compatible with this project's existing worker adapter, and a native from-scratch trainer. These implementations use `torch` and Python, not Hugging Face or downloaded LLM weights.

**The attached `checkpoints/native_small/model.pt` is an INITIALIZED, UNTRAINED checkpoint (step 0).** It is a real, loadable model state, but it has **not learned English, novel writing, instructions, or JSON**. Its text output will be meaningless. This is intentionally not presented as a capable model. The source and checkpoint are included so you own the full model/training/inference path; train on your appropriately licensed corpus to make it useful.

### Run native model standalone

From the project directory (PowerShell or Bash):

```bash
pip install -e '.[test]'
python -m novel_ai.native_model.generate --checkpoint checkpoints/native_small/model.pt --prompt 'Chapter One: The Forgotten Kingdom' --max-new-tokens 50
```

### Start the LOCAL PyTorch inference service

```bash
# Model service; separate terminal, project root
uvicorn novel_ai.native_model.server:app --host 127.0.0.1 --port 8001
```

The `.env.example` is set to `MODEL_BASE_URL=http://localhost:8001/v1`, `MODEL_NAME=native-novel-lm`, and `MODEL_API_KEY=local-only-model-secret`. Start PostgreSQL/Redis, initialize the DB, start API and Celery workers as described above. **Until the checkpoint is trained to follow chapter-writing and JSON-structure instructions, the full novel pipeline will fail at planning or produce unusable content.** That is expected, not an API bug.

### Train from scratch using only your dataset

Create `data/owned_books/` and place your own/licensed `.txt` books there. Training on a corpus with stable long-form narrative style is the first step, but instruction-following and planning need a subsequent explicitly supervised stage. Run:

```bash
python -m novel_ai.native_model.train --data-dir data/owned_books --out checkpoints/native_small/model.pt --steps 10000 --batch-size 8
# Resume, with optimizer state preserved (the initial step-zero checkpoint has no optimizer state)
python -m novel_ai.native_model.train --resume checkpoints/native_small/model.pt --data-dir data/owned_books --out checkpoints/native_small/model.pt --steps 20000
# Test a trained checkpoint
python -m novel_ai.native_model.generate --checkpoint checkpoints/native_small/model.pt --prompt 'Chapter One: The Forgotten Kingdom' --max-new-tokens 300
```

Save every 250 steps by default. Train using a suitably powerful machine and enough data. The checkpoint **holds both model weights and, after actual training, optimizer state**. Use `--context-length`, `--d-model`, `--n-layers`, and `--n-heads` to configure new models. Parameters are set at initialization and not changed when resuming.

To test checkpoint creation independently:

```bash
python -m novel_ai.native_model.init_checkpoint --out checkpoints/native_small/model.pt
pytest tests/test_native_model.py -q
```

### A realistic industrial-grade MODEL roadmap

The included native model is an educational 1.9M-parameter-scale baseline. It proves architecture, training, checkpointing, and integration but **is not a competitive large language model**. Scaling from scratch requires a far larger legally usable corpus, optimized tokenizer, fused operations, distributed FSDP/ZeRO training, activation checkpointing, mixed precision, dynamic batching, KV cache, long-context architecture, evaluation suites, and substantial accelerator time. These optimizations and a pretrained high-quality checkpoint cannot be conjured into the package without actual large-scale training. In particular, quality cannot be inferred from checkpoint loadability or passing unit tests.

For high-quality novel output, prioritize long-context training, character/world-state reasoning datasets, planning and revision objectives, and evaluation on held-out novels after foundational language modeling. Existing `training/sft_lora.py` is an optional **third-party-model** path inherited from the old starter; do not use it for the independent native-model workflow.

## Character-renaming and plot-continuation upgrade (verified with deterministic tests)

After a novel reaches `completed`, retrieve it using `GET /v1/novels/{id}`. Its `state.identity_registry` has stable IDs such as `char_0001`; never guess identity IDs from a display name. Use:

```bash
curl -X POST http://localhost:8080/v1/novels/NOVEL_ID/continue \
  -H 'Authorization: Bearer YOUR_DEV_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"plot":"Kabir and Maya travel to the ancient city and discover a sealed gate that changes their mission.","additional_chapters":2,"renames":[{"character_id":"char_0001","new_name":"Kabir"}]}'
```

This preserves previously published chapter text, rewrites the structured plan and memory to the new current name, normalizes historical summaries for the next prompt, and rejects new chapters containing an exact former-name match before saving. Name collisions, unknown IDs and case-only renames are rejected. If a model emits a forbidden former name, the worker marks the chapter generation as failed/retryable rather than silently publishing incorrect prose. String matching cannot prove full semantic identity consistency (e.g. pronouns, implicit relationships, ambiguous nicknames, events) and the model can still write poor fiction.

Local validation performed: `python -m pytest -q tests/test_continuity.py tests/test_native_model.py tests/test_story.py tests/test_dataset.py`: 15 passed. The full test suite and API import could not run in the packaging environment because `celery` is absent and package download was unavailable. Run `pip install -e '.[test]'` in an online environment before integration testing.

**Model status: Not production-trained.** `checkpoints/native_small/model.pt` is a randomly initialized step-0 checkpoint. Its existence proves the architecture can serialize and load weights, not that the model understands prompts or produces meaningful novels. No private assistant training data or proprietary pre-trained weights are included. The training scripts require user-supplied, legally usable datasets. Do not deploy this as a competitive novel-writing service until model training/evaluation, real inference integration testing, security review and load testing are completed.
