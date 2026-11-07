# The tablerag command in a container:
#   docker run --rm -v "$PWD:/data" ghcr.io/namita-tj/rag index /data/pdfs --db /data/index.db
#   docker run --rm -v "$PWD:/data" ghcr.io/namita-tj/rag ask "Which stage covers 1450 rpm?" --db /data/index.db
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home app
USER app
WORKDIR /data
ENTRYPOINT ["tablerag"]
CMD ["--help"]
