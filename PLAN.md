# Development Plan — Option B: Advanced Hybrid Retrieval & Reranking Benchmark

> Source of truth: `Envint_Global_AI_Intern_Technical_Assignment.pdf`.
> Rule followed: stick to the PDF, no assumptions, **free / open-source tools only** (no paid API).

---

## 1. Chosen option

**Option B — Advanced Hybrid Retrieval & Reranking Benchmark** (page 4).

Scenario: a high-precision compliance engine querying policy & operations manuals, comparing
retrieval strategies on realistic domain queries (exact policy numbers, negative constraints,
semantic nuances).

---

## 2. Technology choices — all named in / allowed by the PDF, all free

| Concern | Choice | PDF basis |
|---|---|---|
| Vector DB | **Qdrant** (local embedded for dev, container service for compose) | Page 2 lists "Qdrant" as an allowed vector DB |
| Embeddings | `sentence-transformers` → `BAAI/bge-small-en-v1.5` | Page 2 "Embeddings-based vector DB" — free HF model |
| Keyword search | `rank_bm25` (BM25) | Option B "BM25 + Vector" |
| Fusion | **Custom Reciprocal Rank Fusion (RRF)** | Objective + Option B "combined via RRF" |
| Reranker | **Cross-encoder `BAAI/bge-reranker-large`** (configurable; `-base` for fast dev) | Option B names "bge-reranker-large" |
| API | FastAPI + Uvicorn | Page 2 "API" service in compose |
| Downstream LLM | Pluggable `LLMClient`. **Default = real free local LLM via Ollama** (e.g. `llama3.2:3b` / `qwen2.5:3b`, open-source, CPU-runnable). Deterministic extractive path kept as offline/CI fallback | Page 2/4 require genuine "LLM responses" & "LLM robustness"; Ollama is free, never a paid API |
| Metrics | Hand-implemented Precision@k, Recall@k, MRR, NDCG@5, p50/p95 latency | Option B "Benchmarking & Pareto Analysis" |
| Resource overhead | `psutil` samples worker CPU%/RSS during benchmark | Option B "resource overhead" |
| Tests | `pytest` (≥15 unit, ≥5 integration) | Page 2 "15 unit tests and 5 integration tests" |
| Observability | Custom middleware logs: step latency, token counts, error rates | Page 2 "custom middleware logs" explicitly allowed |
| CI/CD | **GitHub Actions** (free): lint + run full test suite on push | Page 1 Objective "Production Readiness … CI/CD" |
| Containerization | `docker-compose.yml`: API + Qdrant + test runner | Page 1/2 mandatory |

Nothing above requires payment. Models download once from Hugging Face and run on CPU.

---

## 3. PDF requirement → deliverable traceability

### General requirements (pages 1–2)
- [x] GitHub repo with code, tests, sample dataset, `docker-compose.yml` for one-command run
- [x] `README.md`: architecture diagram, tech trade-offs, setup, reproduction steps, known limitations, **cost-per-query/token breakdown**
- [x] `docker-compose.yml` spins up full env (API, Vector DB, test suite) — see §7 note on "worker/orchestrator"
- [x] CI/CD workflow (GitHub Actions) running lint + tests
- [x] Vector store with explicit payload schema, chunk metadata tagging, embedding versioning
- [x] ≥15 unit tests + ≥5 integration tests — **35 unit + 20 integration = 55 passing**
- [x] LLM guardrails: system/user prompts, strict refusal on missing evidence, output citations
- [x] Observability: step-by-step latency, token counts, error rates
- [x] Retrieved content treated as **evidence/data, never executable instructions**
- [x] Distinguish development vs evaluation data
- [x] Report failures openly (failed tests, limitations, mitigation)
- [x] Metrics: raw structured output (`results/benchmark_*.json`) + concise interpretation (README §5)

### Option B tasks (page 4)
- [x] **3-tier retrieval**: (1) Vector-only, (2) Hybrid BM25+Vector via RRF, (3) Hybrid + cross-encoder rerank
- [x] **Synthetic hard-negative stress dataset**: 13 queries (opposite-meaning keyword matches, fine-grained clause IDs, numeric thresholds)
- [x] Split dataset into **development** (5) and **evaluation** (8) query sets
- [x] **Benchmark matrix**: Precision@k, Recall@k, MRR, NDCG@5, p50/p95 latency (ms), resource overhead
- [x] **Trade-off analysis**: index build time, embedding refresh cost, latency vs relevance gain (Pareto) — README §5
- [x] **LLM robustness**: downstream responses across all 3 strategies vs unsupported queries, conflicting clauses, irrelevant context (`app/eval/llm_robustness.py`)
- [x] **Testing extension**: ranking, RRF, hard negatives, empty results, dependency failures, malformed inputs

---

## 4. Repo layout

```
envint_hybrid_retrieval/
  app/
    config.py            # settings (models, k, env)
    schema.py            # payload / chunk-metadata schemas (pydantic)
    ingest.py            # parse + chunk corpus, embed, upsert to Qdrant
    retrieval/
      vector.py          # tier 1: vector-only
      bm25.py            # BM25 index
      rrf.py             # custom RRF fusion
      rerank.py          # cross-encoder reranker (tier 3)
      engine.py          # unified 3-tier retrieve()
    eval/
      metrics.py         # P@k, R@k, MRR, NDCG@5, latency percentiles
      benchmark.py       # run all tiers over eval set -> matrix + resource overhead (psutil)
      cost.py            # compute-based cost-per-query/token model
    llm/
      client.py          # pluggable LLMClient: Ollama default + extractive fallback
      guardrails.py      # refusal rules, injection-as-data isolation, citations
    obs/
      middleware.py      # latency / token / error logging
    main.py              # FastAPI app
  data/
    corpus/              # sample policy/ops manual docs
    queries/
      dev_queries.json   # development split
      eval_queries.json  # evaluation split (with relevance labels + hard negatives)
  tests/
    unit/                # >=15
    integration/         # >=5
  .github/workflows/ci.yml   # lint + tests (free GitHub Actions)
  docker/Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
  PLAN.md
```

## 5. Build order (incremental, test as we go)
1. Config, schema, sample corpus + query sets (dev/eval, hard negatives)
2. Ingestion → Qdrant (local embedded mode first)
3. Retrieval tiers 1–3 + RRF
4. Metrics + benchmark runner → matrix output
5. LLM client (Ollama + extractive fallback) + guardrails (free local)
6. FastAPI + observability middleware
7. Tests (unit + integration)
8. docker-compose + Dockerfile + GitHub Actions CI
9. README (architecture, trade-offs, cost breakdown, limitations)

## 6. Cost-per-query/token model (free/local → compute cost)

The PDF mandates a cost breakdown. With no paid API, "cost" is **compute**, reported as:
- **Tokens**: input/output token counts per query (from the LLM client; tokenizer count for the extractive path), logged by the observability middleware.
- **Query cost** = measured p50/p95 latency × an assumed hosting rate (e.g. commodity CPU VM $/hr, stated as an assumption). Broken down per stage: embed, BM25, vector search, RRF, rerank, LLM.
- A comparison showing the paid-API equivalent cost (published per-token rates) so the trade-off is explicit, while our actual run cost is $0 API + compute only.

## 7. docker-compose "worker/orchestrator" note (honest scope)

Page 2 lists "worker/orchestrator" among compose services, but that general clause spans all
options — it's aimed at Option C's ingestion pipeline. Option B has no async ingestion worker.
We satisfy the intent by running **ingestion/benchmark as a dedicated one-shot compose service**
(separate from the API) and documenting this reasoning in the README rather than adding a
pointless idle worker.

## 8. Known constraints (honest, per PDF "report failures")
- Docker not installed on dev machine → compose file written & lint-checked, run validated by user later.
- `bge-reranker-large` is CPU-heavy; dev uses `-base`, config switch documented.
- First run downloads HF models + Ollama model (needs internet once); afterwards offline.
- CI runs the extractive LLM fallback (no Ollama in Actions) to keep the pipeline free & fast; Ollama path is exercised locally.
