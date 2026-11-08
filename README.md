# Kondor crawler

Discovers open image fine-tunes / LoRAs, ranks them **per client photo collection** (domain keywords/tags/description × suspicion prior), and writes a Postgres probe queue under monthly screen quotas. Watermarking and GPU probing stay outside this package.

## Product flow

1. Register a client collection (domain text for their photo pack)
2. Crawl HF / Civitai (and optionally GitHub)
3. Score globally (suspicion) and **per collection** (domain relevance)
4. Enqueue top-N into `probe_queue` capped by tier **and** monthly screen quota
5. Probe workers claim rows and write `probe_results`

```bash
kondor-crawl collection upsert \
  --id acme-product \
  --name "Acme product pack" \
  --keywords "product photography,skincare,bottle" \
  --tags "commercial,studio" \
  --description "Commercial product photography of skincare bottles" \
  --tier standard

kondor-crawl run --limit-per-source 50
# or: kondor-crawl enqueue --collection acme-product
```

`embedding` columns on `collections` / `models` are reserved for a later CLIP-style matcher; they are **unused** in scoring today.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run (local)

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql://kondor:kondor@localhost:5432/kondor
kondor-crawl migrate
# register at least one collection (see above) before enqueue does useful work
kondor-crawl run --limit-per-source 50
# optional: --sources huggingface civitai github
# optional: --collection acme-product
```

Scheduling: see [`scripts/cron.example`](scripts/cron.example). Cron / k8s CronJob should call `kondor-crawl run` every few hours, `run --backfill` weekly, and `health` daily.

```bash
docker compose run --rm crawler run
docker compose run --rm crawler run --backfill
docker compose run --rm crawler health
```

Optional: `HF_TOKEN`, `CIVITAI_TOKEN`, `GITHUB_TOKEN`. See `.env.example`.

## Probe team contract

| Table | Writer | Reader | Meaning |
|--------|--------|--------|---------|
| `probe_queue` | crawler | probe workers | Ranked work keyed by `(collection_id, uid)`. Claim with `SKIP LOCKED` or `kondor-crawl claim --worker NAME -n 1 [--collection ID]`. Returns `[{"collection_id","uid"}, …]` ordered by priority |
| `probe_results` | probe workers | crawler | `hit` / `miss` / `error`. Hits promote same-author siblings on the next rescore |
| `collection_scores` | crawler | crawler | Per-collection ranked scores (`RANKER_VERSION`) |
| `screen_usage` | crawler | crawler | Monthly enqueue counts per collection |

```bash
kondor-crawl claim --worker gpu-1 -n 4 --collection acme-product
# → [{"collection_id":"acme-product","uid":"civitai:123"}, ...]
kondor-crawl result 'civitai:123' hit --detail '{"notes":"ok"}'
```

Schema: `run` / `collection` call `create_all` (fine for fresh volumes). Upgrading an existing Alembic-managed DB: `kondor-crawl migrate`.

JSONL export still exists for debugging: `kondor-crawl export -o out/candidates.jsonl [--collection ID]`.

## Commands

| Command | What |
|---------|------|
| `migrate` | Create tables (also done on crawl); includes collections schema |
| `collection upsert/list` | Register client domain descriptors |
| `run` / `crawl` | Incremental crawl + global + per-collection rank + quota enqueue |
| `rescore [--collection]` | Re-score after a ranker change (`RANKER_VERSION`) |
| `enqueue [--collection] [--tier]` | Rebuild `probe_queue` under monthly quotas |
| `eval [--labels] [--collection]` | Precision@K / hit-rate@K |
| `health` | Exit 1 if no successful run in 24h or error rate is high |
| `export` | Ranked JSONL |

## Layout

| Piece | Own it if you… |
|--------|----------------|
| `sources/*` | Add a site (`Source`, `@register`) |
| `rank.py` | Suspicion + domain relevance / `RANKER_VERSION` / tiers |
| `collection.py` | Client collection dataclass |
| `db/` | Schema / storage |
| `probe_queue` + `probe_results` | Team boundary with probe |

## Tiers & monthly quotas

`free` (25) / `standard` (100) / `pro` (500) / `enterprise` (uncapped) — see `rank.TIER_LIMITS`.

Monthly screen quota = the collection's tier limit per calendar month. Quotas count **new enqueues** only (not priority refreshes of already-pending rows).
