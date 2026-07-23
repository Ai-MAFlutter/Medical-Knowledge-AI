FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.docker.txt .

RUN pip install --no-cache-dir \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.docker.txt

COPY rag ./rag
COPY retrieval ./retrieval
COPY monitoring ./monitoring
COPY data ./data
COPY ingestion ./ingestion

CMD ["python", "-m", "rag.test_generator"]