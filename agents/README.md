# Agents

## News + Sentiment Agent (Groq + Selenium)

**Env vars** (repo root `.env`):
- `GROQ_API_KEY` (required untuk analisis AI)
- `GROQ_MODEL` (optional, default: `groq/compound`)

**Usage**
```bash
python agents/news_agent.py --symbol BBCA.JK --max-articles 10 --out agents/output/bbca.json
```

**Optional**
```bash
python agents/news_agent.py --query "IHSG" --detail-limit 3 --headless false
```

## News Agent Server (separate FastAPI server)

Jalankan server terpisah:
```bash
uvicorn agents.server:app --host 0.0.0.0 --port 8010
```

Trigger agent via HTTP:
```bash
curl -X POST http://localhost:8010/news-agent/run \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BBCA.JK","max_articles":10,"detail_limit":3,"headless":true}'
```

Health check:
```bash
curl http://localhost:8010/health
```

Rate limit: 15 requests per 98 seconds per client IP (token bucket).

Queue + polling:
```bash
curl -X POST http://localhost:8010/news-agent/queue \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BBCA.JK","max_articles":10}'

curl http://localhost:8010/news-agent/jobs/<job_id>
```

Server env vars (optional):
- `RATE_LIMIT_WINDOW_SEC` (default 98)
- `RATE_LIMIT_MAX_REQUESTS` (default 15)
- `JOB_TTL_SEC` (default 3600)
- `JOB_CLEANUP_INTERVAL_SEC` (default 300)
- `CACHE_TTL_SEC` (default 600)

Admin clear jobs:
```bash
curl -X POST http://localhost:8010/news-agent/admin/clear-jobs
```
