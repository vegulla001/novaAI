from celery import Celery
from .config import settings
s = settings()
celery_app = Celery("novel_ai", broker=s.redis_url)
celery_app.conf.update(
    task_serializer="json", accept_content=["json"],
    task_acks_late=True, worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": 3600},
    task_routes={"novel_ai.worker.run_novel": {"queue": "novel"}},
    task_track_started=True,
)
