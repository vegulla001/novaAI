import hashlib
import json
import logging
from fastapi import FastAPI, Depends, HTTPException, Header, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from .auth import principal, Principal
from .config import settings
from .db import get_db
from .models import Novel, Chapter, Job, IdempotencyKey
from .schemas import NovelRequest, NovelOut, NovelDetail, ContinueRequest
from .continuity import rename as rename_character, build_continuation_chapter, ContinuityError
from .worker import run_novel

s = settings()
app = FastAPI(title="Novel AI Platform", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",")],
                   allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "Idempotency-Key"])
created = Counter("novel_created_total", "Created novels")

def owned(db: Session, id: str, p: Principal) -> Novel:
    n = db.scalar(select(Novel).where(Novel.id == id, Novel.tenant_id == p.tenant, Novel.user_id == p.user))
    if n is None: raise HTTPException(404, "Novel not found")
    return n

@app.get("/health/live")
def live(): return {"status": "ok"}

@app.get("/health/ready")
def ready(db: Session = Depends(get_db)):
    from sqlalchemy import text
    db.execute(text("SELECT 1"))
    return {"status": "ready"}

@app.post("/v1/novels", response_model=NovelOut, status_code=202)
def create_novel(body: NovelRequest, p: Principal = Depends(principal),
                 db: Session = Depends(get_db), idempotency_key: str | None = Header(default=None)):
    if body.chapters > s.max_chapters or body.words_per_chapter > s.max_words_per_chapter:
        raise HTTPException(422, "Requested generation exceeds server limits")
    if idempotency_key and len(idempotency_key) > 128: raise HTTPException(400, "Idempotency key too long")
    digest = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
    if idempotency_key:
        old = db.scalar(select(IdempotencyKey).where(IdempotencyKey.tenant_id == p.tenant,
                        IdempotencyKey.user_id == p.user, IdempotencyKey.key == idempotency_key))
        if old:
            if old.request_hash != digest: raise HTTPException(409, "Idempotency key reused with different input")
            return owned(db, old.novel_id, p)
    novel = Novel(tenant_id=p.tenant, user_id=p.user, prompt=body.prompt, genre=body.genre,
                  tone=body.tone, chapters_target=body.chapters, words_target=body.words_per_chapter,
                  state={})
    db.add(novel); db.flush()
    db.add(Job(novel_id=novel.id))
    if idempotency_key:
        db.add(IdempotencyKey(tenant_id=p.tenant, user_id=p.user, key=idempotency_key,
                              request_hash=digest, novel_id=novel.id))
    try: db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Concurrent idempotency conflict; retry same key")
    created.inc()
    try: run_novel.delay(novel.id)
    except Exception:
        logging.exception("Initial queue publish failed: recover with reconciler")
    return novel

@app.get("/v1/novels", response_model=list[NovelOut])
def list_novels(p: Principal = Depends(principal), db: Session = Depends(get_db),
                limit: int = 20, offset: int = 0):
    if limit < 1 or limit > 100 or offset < 0: raise HTTPException(422, "Invalid pagination")
    return db.scalars(select(Novel).where(Novel.tenant_id == p.tenant, Novel.user_id == p.user)
                      .order_by(Novel.created_at.desc()).offset(offset).limit(limit)).all()

@app.get("/v1/novels/{novel_id}", response_model=NovelDetail)
def get_novel(novel_id: str, p: Principal = Depends(principal), db: Session = Depends(get_db)):
    n = owned(db, novel_id, p)
    return NovelDetail.model_validate({**NovelOut.model_validate(n).model_dump(), "state": n.state,
            "chapters": list(db.scalars(select(Chapter).where(Chapter.novel_id == n.id)
                                          .order_by(Chapter.number)).all())})

@app.post("/v1/novels/{novel_id}/cancel", response_model=NovelOut)
def cancel_novel(novel_id: str, p: Principal = Depends(principal), db: Session = Depends(get_db)):
    n = owned(db, novel_id, p)
    if n.status not in ("completed", "failed", "cancelled"):
        n.status = "cancelled"
        job = db.scalar(select(Job).where(Job.novel_id == n.id))
        if job: job.status = "cancelled"
        db.commit()
    return n

@app.post("/v1/novels/{novel_id}/resume", response_model=NovelOut)
def resume_novel(novel_id: str, p: Principal = Depends(principal), db: Session = Depends(get_db)):
    n = owned(db, novel_id, p)
    if n.status != "failed": raise HTTPException(409, "Only failed jobs can be resumed")
    n.status = "queued"; n.error = None
    j = db.scalar(select(Job).where(Job.novel_id == novel_id))
    j.status = "queued"; j.lease_owner = None; j.lease_expires_at = None
    db.commit()
    run_novel.delay(novel_id)
    return n

@app.post("/v1/novels/{novel_id}/continue", response_model=NovelOut, status_code=202)
def continue_novel(novel_id: str, body: ContinueRequest, p: Principal = Depends(principal),
                   db: Session = Depends(get_db)):
    # Lock to serialize continuation with competing requests/worker claims.
    n = db.scalar(select(Novel).where(Novel.id == novel_id, Novel.tenant_id == p.tenant,
                Novel.user_id == p.user).with_for_update())
    if n is None: raise HTTPException(404, "Novel not found")
    if n.status != "completed": raise HTTPException(409, "Finish current novel before continuing")
    if n.chapters_target + body.additional_chapters > s.max_chapters:
        raise HTTPException(422, "Chapter limit exceeded")
    if "plan" not in (n.state or {}): raise HTTPException(409, "Novel has no story plan")
    state = dict(n.state)
    try:
        # Already published text is not silently rewritten. Only canon & future prose change.
        if "identity_registry" not in state:
            from .continuity import initialize_registry
            state["identity_registry"] = initialize_registry(state["plan"])
        for instruction in body.renames:
            state = rename_character(state, instruction.character_id, instruction.new_name)
        for ix in range(body.additional_chapters):
            next_number = n.chapters_target + ix + 1
            state["plan"]["chapters"].append(build_continuation_chapter(next_number, body.plot))
    except ContinuityError as exc:
        raise HTTPException(422, str(exc)) from exc
    n.chapters_target += body.additional_chapters
    n.state = state
    n.status = "queued"
    n.error = None
    j = db.scalar(select(Job).where(Job.novel_id == novel_id).with_for_update())
    if j is None: j = Job(novel_id=novel_id); db.add(j)
    j.status = "queued"; j.lease_owner = None; j.lease_expires_at = None
    db.commit()
    try: run_novel.delay(novel_id)
    except Exception: logging.exception("Continuation enqueue failed; reconciler can recover")
    return n

@app.get("/v1/novels/{novel_id}/export")
def export_novel(novel_id: str, p: Principal = Depends(principal), db: Session = Depends(get_db)):
    n = owned(db, novel_id, p)
    chapters = db.scalars(select(Chapter).where(Chapter.novel_id == n.id).order_by(Chapter.number)).all()
    content = "# " + n.title + "\n\n" + "\n\n".join(
        f"## Chapter {c.number}: {c.title}\n\n{c.content}" for c in chapters)
    return PlainTextResponse(content, headers={"Content-Disposition": f'attachment; filename="novel-{n.id}.md"'},
                             media_type="text/markdown; charset=utf-8")

@app.get("/studio", include_in_schema=False)
def studio():
    from pathlib import Path
    return FileResponse(Path(__file__).resolve().parent.parent / "web" / "index.html")
