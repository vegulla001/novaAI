"""Long-running job worker. Durable chapter checkpoints, single-worker lease via Postgres locks."""
import logging
import uuid
from datetime import timedelta
from sqlalchemy import select
from .db import SessionLocal
from .models import Novel, Chapter, Job, utcnow
from .queue import celery_app
from .config import settings
from .story import make_plan, write_chapter, EMPTY_MEMORY
from .continuity import initialize_registry
from .llm import LLM

log = logging.getLogger(__name__)

def claim(novel_id: str, owner: str) -> bool:
    """Select FOR UPDATE guards claiming; stale jobs can be reclaimed."""
    with SessionLocal.begin() as db:
        job = db.scalar(select(Job).where(Job.novel_id == novel_id).with_for_update())
        novel = db.get(Novel, novel_id)
        if job is None or novel is None or novel.status in ("completed", "cancelled", "failed"):
            return False
        now = utcnow()
        if job.lease_expires_at and job.lease_expires_at > now and job.lease_owner != owner:
            return False
        job.status = "running"
        job.lease_owner = owner
        job.lease_expires_at = now + timedelta(seconds=settings().worker_lease_seconds)
        job.attempt += 1
        novel.status = "running"
        return True

def checkpoint(novel_id: str, owner: str, number: int, content: str, summary: str, memory: dict, critique: str):
    with SessionLocal.begin() as db:
        job = db.scalar(select(Job).where(Job.novel_id == novel_id).with_for_update())
        novel = db.get(Novel, novel_id)
        if novel.status == "cancelled": return False
        if job.lease_owner != owner: return False
        existing = db.scalar(select(Chapter).where(Chapter.novel_id == novel_id, Chapter.number == number))
        if not existing:
            title = novel.state["plan"]["chapters"][number - 1]["title"]
            db.add(Chapter(novel_id=novel_id, number=number, title=title,
                           content=content, summary=summary, critique=critique))
        state = dict(novel.state)
        state["memory"] = memory
        novel.state = state
        novel.progress = number
        job.lease_expires_at = utcnow() + timedelta(seconds=settings().worker_lease_seconds)
        if number < novel.chapters_target:
            job.lease_owner = None
            job.lease_expires_at = None
        if number == novel.chapters_target:
            novel.status = job.status = "completed"
            job.lease_owner = None
            job.lease_expires_at = None
        return True

def process_one(novel_id: str, llm=None, owner=None):
    """At most one expensive inference stage per invocation; enqueue next separately."""
    owner = owner or str(uuid.uuid4())
    if not claim(novel_id, owner): return False
    llm = llm or LLM()
    with SessionLocal() as db:
        novel = db.get(Novel, novel_id)
        state = dict(novel.state or {})
        number = novel.progress + 1
        rows = db.scalars(select(Chapter).where(Chapter.novel_id == novel_id).order_by(Chapter.number)).all()
        summaries = [c.summary for c in rows]
        tail = rows[-1].content[-1800:] if rows else ""
        # Access only immutable scalar fields after closing the session.
        if "plan" not in state:
            plan = make_plan(llm, novel)
            with SessionLocal.begin() as write_db:
                j = write_db.scalar(select(Job).where(Job.novel_id == novel_id).with_for_update())
                n = write_db.get(Novel, novel_id)
                if n.status == "cancelled" or j.lease_owner != owner: return False
                n.title = plan["title"][:240]
                n.state = {"plan": plan, "memory": EMPTY_MEMORY, "identity_registry": initialize_registry(plan)}
                j.lease_owner = None
                j.lease_expires_at = None
            return True
        if number > novel.chapters_target:
            with SessionLocal.begin() as write_db:
                n = write_db.get(Novel, novel_id)
                j = write_db.scalar(select(Job).where(Job.novel_id == novel_id).with_for_update())
                n.status = j.status = "completed"
            return False
        content, summary, memory, critique = write_chapter(llm, novel, state, number, summaries, tail)
    return checkpoint(novel_id, owner, number, content, summary, memory, critique)

@celery_app.task(bind=True, name="novel_ai.worker.run_novel", max_retries=3)
def run_novel(self, novel_id):
    # Recovered unacked tasks are safe due to lease + unique (novel_id, number).
    try:
        advanced = process_one(novel_id)
    except Exception as exc:
        log.exception("Novel generation failed for %s", novel_id)
        with SessionLocal.begin() as db:
            job = db.scalar(select(Job).where(Job.novel_id == novel_id).with_for_update())
            if job:
                job.lease_owner = None
                job.lease_expires_at = None
                job.last_error = str(exc)[:1000]
                if self.request.retries >= self.max_retries:
                    job.status = "failed"
                    novel = db.get(Novel, novel_id)
                    if novel.status != "cancelled":
                        novel.status = "failed"
                        novel.error = "Generation failed after retries. Check worker logs."
        raise self.retry(exc=exc, countdown=min(120, 2 ** (self.request.retries + 2)))
    if advanced:
        # Delivery gap between commit and next enqueue can be repaired via reconcile script.
        run_novel.apply_async(args=[novel_id], countdown=1)
