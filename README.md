Envint — Advanced Hybrid Retrieval & Reranking Benchmark (Option B)
A high-precision compliance QA engine over policy & operations manuals that
implements and benchmarks three retrieval strategies, with LLM guardrails,
prompt-injection defence, temporal conflict resolution, observability, and a
one-command Docker environment.
Quick links (when the app is running): 🚀 Web UI ·
📚 Swagger API Docs ·
❤️ Health Check ·
📈 Metrics
These localhost links work on the computer running the app; they are not public deployment links.
> **Everything here is free / open-source.** No paid API is ever called. Models
> download once from Hugging Face and run on CPU; the optional LLM runs locally
> via Ollama. Tests and CI run fully offline with deterministic "fake" backends,
> so the whole suite costs **$0**.
---
1. Architecture
```
                         ┌──────────────────────────────────────────┐
   HTTP (FastAPI)        │               RetrievalEngine             │
  POST /query ──────────►│                                           │
                         │   ┌───────────┐     ┌───────────┐         │
                         │   │  Vector   │     │   BM25    │         │
                         │   │  (Qdrant) │     │ (keyword) │         │
                         │   └─────┬─────┘     └─────┬─────┘         │
   Tier 1: vector ───────────────► │                 │              │
                         │         └───────┬─────────┘              │
   Tier 2: hybrid ──────────────►  Custom RRF fusion                │
                         │                 │                        │
   Tier 3: rerank ──────────────►  Cross-encoder reranker           │
                         │                 │ (bge-reranker)         │
                         │                 ▼                        │
                         │        top-k RetrievedChunks             │
                         └─────────────────┬────────────────────────┘
                                           ▼
                         ┌──────────────────────────────────────────┐
                         │                Guardrails                 │
                         │  • refuse if evidence doesn't support      │
                         │  • isolate context as UNTRUSTED data       │
                         │  • detect + resolve temporal conflicts     │
                         └─────────────────┬────────────────────────┘
                                           ▼
                         ┌──────────────────────────────────────────┐
                         │  LLM client: Ollama (local) | extractive  │
                         │  → answer + citations + conflict notes     │
                         └──────────────────────────────────────────┘

  Observability middleware wraps every request: per-step latency, token counts,
  error rate. Benchmark runner scores all 3 tiers → results/benchmark_*.json.
```
Request flow: `POST /query` → engine retrieves with the chosen tier →
guardrails enforce refusal / injection isolation / conflict resolution → LLM
client produces a cited answer → middleware records latency, tokens, errors.
---
2. The three retrieval tiers
Tier	Strategy	What it does	Strength
1	`vector`	Embed query, nearest-neighbour search in Qdrant (cosine)	Semantic meaning, paraphrases
2	`hybrid`	BM25 + vector, combined with custom RRF	Adds exact keywords, clause IDs, numbers
3	`rerank`	Tier-2 candidates re-scored by a cross-encoder	Highest precision on hard negatives
Custom RRF (app/retrieval/rrf.py) ignores the
incompatible raw scores of BM25 vs vectors and fuses by rank: a chunk at rank
`r` earns `1/(k+r)` from each list. A chunk ranked highly by both retrievers
wins. `k=60` (the standard value) softens the top-rank dominance.
---
3. Technology choices & trade-offs
Concern	Choice	Why / trade-off
Vector DB	Qdrant (in-memory for dev/tests, container for compose)	Free, fast, runs in-process so tests need no server
Embeddings	`BAAI/bge-small-en-v1.5` (384-dim)	Small, CPU-friendly, strong quality/size trade-off
Keyword	`rank_bm25`	Pure-Python, zero infra; the literal counterpart to vectors
Fusion	hand-written RRF	Transparent, score-scale agnostic, no tuning needed
Reranker	`BAAI/bge-reranker-base` (PDF names `-large`)	`-base` is far faster on CPU; switch via `RERANK_MODEL`
LLM	Ollama local (`llama3.2:3b`) default, extractive fallback	Genuine LLM answers for free; extractive path is deterministic & offline
API	FastAPI + Uvicorn	Async, typed, auto-docs
Metrics	hand-implemented	Transparent maths, no black-box
Observability	custom middleware logs	Explicitly allowed by the brief; no external service
CI/CD	GitHub Actions	Free; runs the offline suite on every push
Offline test backends. Setting `EMBED_BACKEND=fake` / `RERANK_BACKEND=fake` /
`LLM_BACKEND=extractive` swaps the ML models for deterministic, dependency-free
stand-ins (hash embeddings, lexical reranker, sentence-extraction answerer). This
is what makes the test suite and CI run anywhere, instantly, for free.
---
4. Setup & reproduction
Option A — Docker (one command)
```bash
# Start Qdrant and the API (first build may download model files)
docker compose up --build api

# In a second terminal: full test suite using offline backends
docker compose --profile test run --rm tests

# Retrieval benchmark
docker compose --profile benchmark run --rm benchmark
```
Option B — Local Python
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload

# Open http://localhost:8000/ in your browser

# Run tests (Windows PowerShell: set offline backends first)
$env:EMBED_BACKEND="fake"; $env:RERANK_BACKEND="fake"; $env:LLM_BACKEND="extractive"
python -m pytest -q

# Run the benchmark and robustness checks
python -m app.eval.benchmark --split eval
python -m app.eval.llm_robustness
```
Try it
🚀 Web UI — click to open — ask questions,
switch strategy, and see answers, citations, conflict banners, the
injection-defence shield, and latency/token stats. (Served from
app/static/index.html; no build step.)
📚 Swagger UI — click to open — interactive API docs.
❤️ Health check — verify the API is running.
📈 Metrics — view observability metrics.
curl:
```bash
curl -s localhost:8000/query -H "content-type: application/json" \
  -d '{"query":"how often must passwords be rotated?","strategy":"rerank"}'
```
Windows PowerShell alternative:
```powershell
$body = @{
  query = "how often must passwords be rotated?"
  strategy = "rerank"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/query" `
  -Method Post -ContentType "application/json" -Body $body
```
Optional: real local LLM (still free)
```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.2:3b
LLM_BACKEND=auto uvicorn app.main:app   # uses Ollama if reachable, else extractive
```
---
5. Benchmark results & interpretation
Comparative matrix on the evaluation split (14 queries: 12 with relevance
labels and 2 unsupported queries). These measurements were recorded with the real
local embedding and reranking setup; they are environment-specific, while CI uses
deterministic offline backends. Raw JSON: `results/benchmark_eval.json`.
strategy	Precision@5	Recall@5	MRR	NDCG@5	p50 (ms)	p95 (ms)	peak RSS (MB)
vector	0.2167	1.000	0.9167	0.9318	19.23	29.75	556.3
hybrid	0.2167	1.000	0.9167	0.9385	16.42	22.46	538.3
rerank	0.2167	1.000	0.9583	0.9692	872.96	952.24	1151.1
Interpretation (quality vs. latency).
Ranking quality: reranking has the highest recorded MRR (0.9583) and
NDCG@5 (0.9692). Hybrid improves NDCG@5 over vector-only retrieval
(0.9385 vs. 0.9318).
Latency: vector and hybrid p50 latency are 19.23 ms and 16.42 ms.
Reranking raises p50 to 872.96 ms and p95 to 952.24 ms in this run, so the
cross-encoder has a substantial query-time cost.
Precision and recall: Precision@5 is 0.2167 and Recall@5 is 1.0 for all
three strategies in this run. The corpus and evaluation set are small and
synthetic, so these results are preliminary.
Index/refresh cost trade-offs. The recorded index build took approximately
13.87 seconds for the current corpus and environment. Build time depends on the
embedding model, hardware, and corpus size. Re-embedding is only required
when the embedding model changes — each chunk stores its `embedding_version`
(see embedding versioning handling below), so a version bump is the signal to
re-index. The reranker needs no index (it scores at query time).
Embedding versioning handling. On every index build the engine stamps each
chunk with the currently configured model's `embedding_version`
(engine.py). This guarantees the stored version
always matches the vectors that actually exist, so you can never mix vectors from
two different models; changing the model and rebuilding re-embeds and re-stamps
automatically.
---
6. Cost-per-query / token breakdown
We call no paid API, so API cost is $0. The real cost is compute time,
reported by app/eval/cost.py:
```
compute_cost_usd   = latency_hours × COMPUTE_USD_PER_HOUR   (assumed $0.10/hr CPU VM)
api_cost_usd       = 0.0                                    (free local models)
paid_equivalent_usd= tokens priced at a typical hosted rate (comparison only)
```
Example (extractive backend, a short query ≈ 10 in / 20 out tokens, ~1 ms):
`compute_cost_usd` ≈ `1ms/3.6e6 × $0.10` ≈ $2.8e-8
`api_cost_usd` = $0
`paid_equivalent_usd` ≈ $1.4e-5 (what a hosted per-token API would charge)
Token counts per request are returned in the `/query` response (`token_usage`)
and logged by the observability middleware.
---
7. Robustness features (how the brief is satisfied)
Hard-negative dataset — data/queries/ has 19 queries
split dev (5) / eval (14), including opposite-meaning keyword matches,
fine-grained clause IDs, numeric thresholds, unsupported questions, and a
temporal conflict.
Refusal on missing evidence — `is_supported()` refuses when no retrieved
chunk shares meaningful words with the query (e.g. "remote work policy").
Prompt-injection defence — 5 malicious snippets are embedded in the corpus
(`SEC-3.2`, `RET-3.1`, `OPS-2.2`, `AC-2.1`, `TE-2.1`). Retrieved text is framed
as untrusted data in a delimited block the LLM is told to distrust; the
extractive backend is immune by construction (it only copies sentences). We
verify the answer never emits the injected "100% compliant" string.
Temporal conflict resolution — when two clauses in the same document assert
different numbers (customer retention `7 years` 2024 vs `3 years` 2022), the
engine surfaces the conflict and resolves it to the most recent clause.
Observability — per-step latency, token counts, and a running error rate
(`/metrics`), via custom middleware.
---
8. Testing
```
59 tests passed in the recorded Docker test run: 36 unit + 23 integration
(brief requires ≥15 unit and ≥5 integration).
```
Coverage includes: RRF correctness, all metrics, BM25, guardrails
(injection/refusal/conflict), embeddings, cost, schema validation, the 3-tier
engine, the API (happy path, refusal, 422s, injection isolation, temporal
truth), the benchmark runner, malformed inputs (missing file, bad JSON,
missing required field), empty results (empty corpus → no hits → refusal),
rate-limit failures (LLM HTTP 429 → typed `LLMRateLimitError`),
embedding-version handling (stale version re-stamped on build), and
dependency failures (Ollama outage fallback, vector-store timeout → HTTP 500).
Run: `pytest -q` (offline backends set in tests/conftest.py).
---

<img width="952" height="1057" alt="Screenshot 2026-10-09 171951" src="https://github.com/user-attachments/assets/b1085893-7e40-491f-a32a-3b1b33254e07" />

<img width="1917" height="992" alt="Screenshot 2026-10-09 172112" src="https://github.com/user-attachments/assets/b67e0c76-e75b-436c-b432-c1b2b7117fa1" />

<img width="1917" height="1031" alt="Screenshot 2026-10-09 173349" src="https://github.com/user-attachments/assets/a62d9629-8947-46c4-89f7-269a90a53193" />



9. Known limitations (reported openly, per the brief)
Small synthetic evaluation set — the reported benchmark uses 14 eval
queries (12 labelled); larger independently authored evaluation sets are
needed before drawing production conclusions.
Hardware-specific benchmark numbers — the displayed measurements use real
local embedding/reranking models on this environment. CI uses deterministic
fake backends, so CI timings should not be compared directly to this table.
Docker validation — the Docker test profile was run successfully with
59 tests passing. Startup time and real-model performance still depend on
local hardware and model-cache state.
`bge-reranker-large` (named in the PDF) is heavy on CPU; the default is the
same-family `-base`. Switch with `RERANK_MODEL=BAAI/bge-reranker-large`.
Corpus is small and pre-chunked — Option B is about retrieval, so we use
a labelled JSON corpus for exact ground truth. Parsing raw PDF/DOCX/PPTX is
the focus of Option C and is intentionally out of scope here.
Refusal/support is a lexical heuristic — robust for this corpus; a
production system would add a semantic relevance threshold.
---
10. Repo layout
```
app/
  config.py            settings (env-overridable)
  schema.py            payload / chunk-metadata schemas
  embeddings.py        real (HF) + fake (offline) embedders
  ingest.py            load + validate corpus
  retrieval/           vector · bm25 · rrf · rerank · engine (3 tiers)
  eval/                metrics · benchmark · cost · llm_robustness
  llm/                 guardrails · client (Ollama + extractive)
  obs/                 observability middleware
  main.py              FastAPI app
data/
  corpus/corpus.json   policy/ops chunks (clause IDs, conflicts, injections)
  queries/             dev_queries.json · eval_queries.json (gold labels)
tests/
  unit/  integration/  (55 tests)
docker/Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```
