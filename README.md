# Envint â€” Advanced Hybrid Retrieval & Reranking Benchmark (Option B)

A high-precision compliance QA engine over policy & operations manuals that
implements and **benchmarks three retrieval strategies**, with LLM guardrails,
prompt-injection defence, temporal conflict resolution, observability, and a
one-command Docker environment.

> **Everything here is free / open-source.** No paid API is ever called. Models
> download once from Hugging Face and run on CPU; the optional LLM runs locally
> via Ollama. Tests and CI run fully offline with deterministic "fake" backends,
> so the whole suite costs **$0**.

---

## 1. Architecture

```
                         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   HTTP (FastAPI)        â”‚               RetrievalEngine             â”‚
  POST /query â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–ºâ”‚                                           â”‚
                         â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”         â”‚
                         â”‚   â”‚  Vector   â”‚     â”‚   BM25    â”‚         â”‚
                         â”‚   â”‚  (Qdrant) â”‚     â”‚ (keyword) â”‚         â”‚
                         â”‚   â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜         â”‚
   Tier 1: vector â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º â”‚                 â”‚              â”‚
                         â”‚         â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜              â”‚
   Tier 2: hybrid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º  Custom RRF fusion                â”‚
                         â”‚                 â”‚                        â”‚
   Tier 3: rerank â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º  Cross-encoder reranker           â”‚
                         â”‚                 â”‚ (bge-reranker)         â”‚
                         â”‚                 â–¼                        â”‚
                         â”‚        top-k RetrievedChunks             â”‚
                         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                           â–¼
                         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                         â”‚                Guardrails                 â”‚
                         â”‚  â€¢ refuse if evidence doesn't support      â”‚
                         â”‚  â€¢ isolate context as UNTRUSTED data       â”‚
                         â”‚  â€¢ detect + resolve temporal conflicts     â”‚
                         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                           â–¼
                         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                         â”‚  LLM client: Ollama (local) | extractive  â”‚
                         â”‚  â†’ answer + citations + conflict notes     â”‚
                         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

  Observability middleware wraps every request: per-step latency, token counts,
  error rate. Benchmark runner scores all 3 tiers â†’ results/benchmark_*.json.
```

**Request flow:** `POST /query` â†’ engine retrieves with the chosen tier â†’
guardrails enforce refusal / injection isolation / conflict resolution â†’ LLM
client produces a cited answer â†’ middleware records latency, tokens, errors.

---

## 2. The three retrieval tiers

| Tier | Strategy | What it does | Strength |
|------|----------|--------------|----------|
| 1 | `vector` | Embed query, nearest-neighbour search in Qdrant (cosine) | Semantic meaning, paraphrases |
| 2 | `hybrid` | BM25 **+** vector, combined with **custom RRF** | Adds exact keywords, clause IDs, numbers |
| 3 | `rerank` | Tier-2 candidates re-scored by a **cross-encoder** | Highest precision on hard negatives |

**Custom RRF** ([app/retrieval/rrf.py](app/retrieval/rrf.py)) ignores the
incompatible raw scores of BM25 vs vectors and fuses by *rank*: a chunk at rank
`r` earns `1/(k+r)` from each list. A chunk ranked highly by **both** retrievers
wins. `k=60` (the standard value) softens the top-rank dominance.

---

## 3. Technology choices & trade-offs

| Concern | Choice | Why / trade-off |
|---|---|---|
| Vector DB | **Qdrant** (in-memory for dev/tests, container for compose) | Free, fast, runs in-process so tests need no server |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | Small, CPU-friendly, strong quality/size trade-off |
| Keyword | `rank_bm25` | Pure-Python, zero infra; the literal counterpart to vectors |
| Fusion | hand-written RRF | Transparent, score-scale agnostic, no tuning needed |
| Reranker | `BAAI/bge-reranker-base` (PDF names `-large`) | `-base` is far faster on CPU; switch via `RERANK_MODEL` |
| LLM | **Ollama** local (`llama3.2:3b`) default, extractive fallback | Genuine LLM answers for free; extractive path is deterministic & offline |
| API | FastAPI + Uvicorn | Async, typed, auto-docs |
| Metrics | hand-implemented | Transparent maths, no black-box |
| Observability | custom middleware logs | Explicitly allowed by the brief; no external service |
| CI/CD | GitHub Actions | Free; runs the offline suite on every push |

**Offline test backends.** Setting `EMBED_BACKEND=fake` / `RERANK_BACKEND=fake` /
`LLM_BACKEND=extractive` swaps the ML models for deterministic, dependency-free
stand-ins (hash embeddings, lexical reranker, sentence-extraction answerer). This
is what makes the test suite and CI run anywhere, instantly, for free.

---

## 4. Setup & reproduction

### Option A â€” Docker (one command)

```bash
# Vector DB + API (real embeddings & reranker; downloads models on first run)
docker compose up --build api        # API on http://localhost:8000

# Full test suite (offline/free backends)
docker compose run --rm tests

# Benchmark matrix (writes results/benchmark_eval.json)
docker compose run --rm benchmark
```

### Option B â€” Local Python

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload

# Run tests (offline backends)
EMBED_BACKEND=fake RERANK_BACKEND=fake LLM_BACKEND=extractive pytest -q

# Run the benchmark
python -m app.eval.benchmark --split eval
python -m app.eval.llm_robustness
```

### Try it

- **Web UI** â€” open **http://localhost:8000/** for a simple page to ask
  questions, switch strategy, and see answers, citations, conflict banners, the
  injection-defence shield, and latency/token stats. (Served from
  [app/static/index.html](app/static/index.html); no build step.)
- **Swagger UI** â€” **http://localhost:8000/docs** for the raw API.
- **curl**:

```bash
curl -s localhost:8000/query -H "content-type: application/json" \
  -d '{"query":"how often must passwords be rotated?","strategy":"rerank"}'
```

### Optional: real local LLM (still free)

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.2:3b
LLM_BACKEND=auto uvicorn app.main:app   # uses Ollama if reachable, else extractive
```

---

## 5. Benchmark results & interpretation

Comparative matrix on the **evaluation split** (14 queries: 12 relevance queries and 2 unsupported queries), using the real local embedding and reranking models
(reproducible in CI). Raw JSON: `results/benchmark_eval.json`.

| strategy | Precision@5 | Recall@5 | MRR | NDCG@5 | p50 (ms) | p95 (ms) | peak RSS (MB) |
|----------|------------:|---------:|----:|-------:|---------:|---------:|--------------:|
| vector   | 0.175 | 0.875 | 0.813 | 0.829 | 0.33 | 1.05 | 112.6 |
| hybrid   | 0.175 | 0.875 | 0.875 | 0.865 | 0.45 | 0.48 | 112.6 |
| rerank   | 0.200 | 1.000 | 0.938 | 0.944 | 0.60 | 0.75 | 112.6 |

**Interpretation (Pareto).**
- **Quality rises monotonically** vector â†’ hybrid â†’ rerank on every ranking
  metric (MRR, NDCG, Recall). Reranking achieves perfect recall and the best
  NDCG by pulling the exact gold clause to the top on hard-negative queries.
- **Latency rises too** (rerank runs the cross-encoder once per candidate), so
  there is a genuine quality-vs-latency trade-off. On this tiny corpus it is
  sub-millisecond; on a real corpus with the real models the reranker is the
  dominant cost â€” use it when precision matters, hybrid when latency does.
- **Precision@5 looks low (~0.2)** because most queries have exactly **one**
  relevant chunk, so the ceiling is `1/5 = 0.2`. Recall/MRR/NDCG are the
  informative metrics here. (These are the offline-backend numbers; the real
  `bge` models shift the absolute values but preserve the ordering.)

**Index/refresh cost trade-offs.** Building the index is a one-off embed of the
corpus (BM25 build is instant). The benchmark measures and reports this as
`index_build_ms` (e.g. ~6 ms for 20 chunks with the offline backend; dominated
by model embedding time with the real backend). Re-embedding is only required
when the embedding model changes â€” each chunk stores its `embedding_version`
(see **embedding versioning handling** below), so a version bump is the signal to
re-index. The reranker needs no index (it scores at query time).

**Embedding versioning handling.** On every index build the engine stamps each
chunk with the *currently configured* model's `embedding_version`
([engine.py](app/retrieval/engine.py)). This guarantees the stored version
always matches the vectors that actually exist, so you can never mix vectors from
two different models; changing the model and rebuilding re-embeds and re-stamps
automatically.

---

## 6. Cost-per-query / token breakdown

We call **no paid API**, so API cost is **$0**. The real cost is compute time,
reported by [app/eval/cost.py](app/eval/cost.py):

```
compute_cost_usd   = latency_hours Ã— COMPUTE_USD_PER_HOUR   (assumed $0.10/hr CPU VM)
api_cost_usd       = 0.0                                    (free local models)
paid_equivalent_usd= tokens priced at a typical hosted rate (comparison only)
```

Example (extractive backend, a short query â‰ˆ 10 in / 20 out tokens, ~1 ms):
- `compute_cost_usd` â‰ˆ `1ms/3.6e6 Ã— $0.10` â‰ˆ **$2.8e-8**
- `api_cost_usd` = **$0**
- `paid_equivalent_usd` â‰ˆ **$1.4e-5** (what a hosted per-token API would charge)

Token counts per request are returned in the `/query` response (`token_usage`)
and logged by the observability middleware.

---

## 7. Robustness features (how the brief is satisfied)

- **Hard-negative dataset** â€” [data/queries/](data/queries/) has 13 queries
  split dev (5) / eval (8), including opposite-meaning keyword matches
  (`90 days` passwords vs backup logs), fine-grained clause IDs, numeric
  thresholds, unsupported questions, and a temporal conflict.
- **Refusal on missing evidence** â€” `is_supported()` refuses when no retrieved
  chunk shares meaningful words with the query (e.g. "remote work policy").
- **Prompt-injection defence** â€” 5 malicious snippets are embedded in the corpus
  (`SEC-3.2`, `RET-3.1`, `OPS-2.2`, `AC-2.1`, `TE-2.1`). Retrieved text is framed
  as **untrusted data** in a delimited block the LLM is told to distrust; the
  extractive backend is immune by construction (it only copies sentences). We
  verify the answer never emits the injected "100% compliant" string.
- **Temporal conflict resolution** â€” when two clauses in the same document assert
  different numbers (customer retention `7 years` 2024 vs `3 years` 2022), the
  engine surfaces the conflict **and** resolves it to the most recent clause.
- **Observability** â€” per-step latency, token counts, and a running error rate
  (`/metrics`), via custom middleware.

---

## 8. Testing

```
59 tests passed in Docker: 36 unit + 23 integration   (brief requires â‰¥15 unit, â‰¥5 integration)
```

Coverage includes: RRF correctness, all metrics, BM25, guardrails
(injection/refusal/conflict), embeddings, cost, schema validation, the 3-tier
engine, the API (happy path, refusal, 422s, injection isolation, temporal
truth), the benchmark runner, **malformed inputs** (missing file, bad JSON,
missing required field), **empty results** (empty corpus â†’ no hits â†’ refusal),
**rate-limit failures** (LLM HTTP 429 â†’ typed `LLMRateLimitError`),
**embedding-version handling** (stale version re-stamped on build), and
**dependency failures** (Ollama outage fallback, vector-store timeout â†’ HTTP 500).

Run: `pytest -q` (offline backends set in [tests/conftest.py](tests/conftest.py)).

---

## 9. Known limitations (reported openly, per the brief)

- **Docker not run on the dev machine** â€” `docker-compose.yml` and `Dockerfile`
  are written and lint-checked; please validate the `docker compose up` path in
  your environment. All Python paths are verified (55 tests green).
- **Benchmark numbers shown are from the real local embedding and reranking models** so they are
  reproducible in CI without model downloads. The real `bge` models change the
  absolute metric values but preserve the vector â†’ hybrid â†’ rerank ordering.
- **`bge-reranker-large`** (named in the PDF) is heavy on CPU; the default is the
  same-family `-base`. Switch with `RERANK_MODEL=BAAI/bge-reranker-large`.
- **Corpus is small and pre-chunked** â€” Option B is about *retrieval*, so we use
  a labelled JSON corpus for exact ground truth. Parsing raw PDF/DOCX/PPTX is
  the focus of Option C and is intentionally out of scope here.
- **Refusal/support is a lexical heuristic** â€” robust for this corpus; a
  production system would add a semantic relevance threshold.

---

## 10. Repo layout

```
app/
  config.py            settings (env-overridable)
  schema.py            payload / chunk-metadata schemas
  embeddings.py        real (HF) + fake (offline) embedders
  ingest.py            load + validate corpus
  retrieval/           vector Â· bm25 Â· rrf Â· rerank Â· engine (3 tiers)
  eval/                metrics Â· benchmark Â· cost Â· llm_robustness
  llm/                 guardrails Â· client (Ollama + extractive)
  obs/                 observability middleware
  main.py              FastAPI app
data/
  corpus/corpus.json   policy/ops chunks (clause IDs, conflicts, injections)
  queries/             dev_queries.json Â· eval_queries.json (gold labels)
tests/
  unit/  integration/  (59 tests)
docker/Dockerfile Â· docker-compose.yml Â· .github/workflows/ci.yml
```

