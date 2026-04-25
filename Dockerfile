FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/
COPY eval/ eval/
COPY knowledge_base/ knowledge_base/
COPY viz/ viz/
COPY .env.example .env.example

RUN mkdir -p /app/photos /app/knowledge_base /app/eval/results /home/user \
    && chown -R 1000:1000 /app /home/user

ENV HOME=/home/user

EXPOSE 7860
USER 1000

CMD ["sh", "-c", "python scripts/fetch_assets.py && uvicorn api.server:app --host 0.0.0.0 --port 7860"]
