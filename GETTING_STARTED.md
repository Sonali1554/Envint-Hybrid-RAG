# Getting Started

How to run the project from scratch, plus the tasks still left to do.

---

## Part 1 — Run the project

You can run it **two ways**. Docker is the one-command path the assignment asks
for; the local path is handy for quick iteration.

### Prerequisites
- **Docker path:** Docker Desktop installed and running.
- **Local path:** Python 3.11+ installed.

---

### Option A — Docker (one command, matches the assignment)

```bash
# From the project root (where docker-compose.yml lives)

# 1. Start the Vector DB (Qdrant) + API. First run downloads the bge models.
docker compose up --build api
#    → API live at http://localhost:8000   (UI at /, Swagger at /docs)

# 2. In a second terminal: run the full test suite (offline, free backends)
docker compose run --rm tests

# 3. Run the benchmark matrix (writes results/benchmark_eval.json)
docker compose run --rm benchmark

# 4. Stop everything
docker compose down
```

---

### Option B — Local Python (fast iteration)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run the API with REAL models (best answer quality; downloads bge once)
uvicorn app.main:app --reload
#     → open http://localhost:8000/

# 3b. OR run the API in FAST offline mode (no downloads, instant start)
#     (quality is lower, but every feature works)
#   PowerShell:
$env:EMBED_BACKEND="fake"; $env:RERANK_BACKEND="fake"; $env:LLM_BACKEND="extractive"
uvicorn app.main:app --reload
#   bash:
EMBED_BACKEND=fake RERANK_BACKEND=fake LLM_BACKEND=extractive uvicorn app.main:app --reload

# 4. Run tests (always use the offline backends)
#   PowerShell:
$env:EMBED_BACKEND="fake"; $env:RERANK_BACKEND="fake"; $env:LLM_BACKEND="extractive"; pytest -q
#   bash:
EMBED_BACKEND=fake RERANK_BACKEND=fake LLM_BACKEND=extractive pytest -q

# 5. Run the benchmark and robustness probe
python -m app.eval.benchmark --split eval
python -m app.eval.llm_robustness
```

---

### How to actually *use* it once running

- **Web UI:** http://localhost:8000/ — type a question, pick a strategy
  (Vector / Hybrid / Rerank), see the answer, citations, conflict + injection
  banners, latency and token counts. Click an example chip to try quickly.
- **Swagger UI:** http://localhost:8000/docs — the raw API.
- **curl:**
  ```bash
  curl -s localhost:8000/query -H "content-type: application/json" \
    -d '{"query":"how long must customer records be retained?","strategy":"rerank"}'
  ```

**Good questions to demo the robustness features:**
| Question | What it shows |
|---|---|
| `how often must passwords be rotated?` | normal retrieval + citations |
| `how long must customer records be retained?` | temporal conflict → resolved to 7 years |
| `summarise the audit log review process` | injection detected & neutralized |
| `what is the company remote work policy?` | strict refusal (no evidence) |

---

## Part 2 — Tasks left to do

Everything required by the PDF is **built and passing (59 tests)**. What remains
is validation, optional polish, and submission.

### Must do before submitting
- [ ] **Validate the Docker path** — run `docker compose up --build api` on a
      machine with Docker and confirm the API comes up at :8000. (Could not be
      run in the dev environment; files are written and lint-checked.)
- [ ] **Push to a GitHub repository** — the PDF requires a repo containing the
      code, tests, sample dataset, and `docker-compose.yml`.

### Optional polish (nice-to-have, not required)
- [ ] Run the benchmark with the **real `bge` models** and paste the real
      numbers into README §5 (replace the offline-backend numbers).
- [ ] Add an **Ollama service** to `docker-compose.yml` so the real local LLM
      runs end-to-end in containers (currently defaults to the extractive LLM).
- [ ] Expand the corpus beyond the 20 sample chunks if you want richer demos.

### Already complete (no action needed)
- [x] 3-tier retrieval (vector / hybrid-RRF / rerank)
- [x] Hard-negative dataset (13 queries) with dev/eval split
- [x] Benchmark matrix: P@k, R@k, MRR, NDCG@5, p50/p95, resource overhead, index build time
- [x] Pareto / trade-off analysis (README §5)
- [x] LLM robustness across all 3 strategies
- [x] Guardrails: refusal, injection isolation, citations, temporal conflicts
- [x] Observability: step latency, token counts, error rate
- [x] Vector store schema, metadata tagging, embedding-version handling
- [x] ≥15 unit + ≥5 integration tests (have 36 + 23)
- [x] docker-compose, Dockerfile, GitHub Actions CI
- [x] README with architecture, trade-offs, cost-per-query/token, limitations
- [x] Bonus: web UI + Swagger
