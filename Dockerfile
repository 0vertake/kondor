FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY alembic ./alembic

RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 kondor \
    && chown -R kondor:kondor /app

USER kondor
ENTRYPOINT ["kondor-crawl"]
CMD ["run"]
