"""Run periodically as a single scheduled task; re-publish orphaned jobs."""
from sqlalchemy import select
from .db import SessionLocal
from .models import Job, Novel, utcnow
from .worker import run_novel

def reconcile():
    with SessionLocal() as db:
        jobs = db.execute(select(Job, Novel).join(Novel, Job.novel_id == Novel.id)
                          .where(Novel.status.in_(["queued", "running"]))).all()
    for job, novel in jobs:
        if not job.lease_expires_at or job.lease_expires_at < utcnow():
            run_novel.delay(novel.id)
    return len(jobs)

if __name__ == "__main__":
    print("Checked", reconcile(), "in-progress jobs")
