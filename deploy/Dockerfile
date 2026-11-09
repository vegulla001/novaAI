FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY pyproject.toml ./
COPY novel_ai ./novel_ai
RUN pip install --no-cache-dir .
COPY web ./web
RUN useradd --create-home appuser
USER appuser
CMD ["uvicorn", "novel_ai.api:app", "--host", "0.0.0.0", "--port", "8080"]
