# Conversational SHL Assessment Recommender

A stateless FastAPI service for the SHL AI Intern take-home assignment.

It exposes exactly the required endpoints:

- `GET /health` -> `{"status": "ok"}`
- `POST /chat` -> strict response schema with `reply`, `recommendations`, `end_of_conversation`

The agent supports clarification, catalog-grounded recommendation, refinement, comparison and refusal. It never returns URLs outside `data/catalog.json`.

## 1. Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Build the full SHL catalog

The repo includes a small `data/catalog.json` only so tests and local startup work immediately. Before submission, replace it with the full scraped catalog:

```bash
python scripts/scrape_catalog.py --out data/catalog.json
```

The scraper targets SHL Individual Test Solutions via the product catalog pagination and enriches each product page with description, job levels, languages, duration and test type.

## 3. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer with SQL and stakeholder communication"}]}'
```

## 4. Run tests

```bash
pytest -q
python scripts/eval_traces.py
# If SHL gives public trace JSON files:
python scripts/eval_traces.py path/to/public_traces
```

## 5. Deploy on Render

1. Push this folder to GitHub.
2. Create a new Render Web Service.
3. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Set env var if needed: `CATALOG_PATH=data/catalog.json`
5. Submit the Render base URL as your public endpoint URL.

Your submitted endpoint URL will look like:

```text
https://your-service-name.onrender.com
```

The evaluator will call:

```text
https://your-service-name.onrender.com/health
https://your-service-name.onrender.com/chat
```

## 6. Why this design scores safely

- Strict Pydantic request/response schema.
- `messages` length capped at 8 to match evaluator turn cap.
- Recommendations validated against local catalog URLs.
- Empty recommendations on clarification or refusal.
- Hybrid BM25 + TF-IDF + metadata retrieval for higher Recall@10.
- Stateless: every request rebuilds intent from full conversation history.
