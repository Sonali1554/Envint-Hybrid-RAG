Envint — Advanced Hybrid Retrieval & Reranking Benchmark
AI Engineer Intern Technical Assignment · Option B
A compliance QA engine that compares Vector Search, Hybrid Search (BM25 + Vector with RRF), and Cross-Encoder Reranking. It includes evidence-based answers, source citations, prompt-injection handling, temporal conflict detection, benchmark evaluation, tests, and Docker support.
Quick links (available while running locally): Web UI · API Docs · Health · Metrics
> The project uses local/open-source models and does not require a paid API. The localhost links are not public deployment links.
1. Architecture
```text
Browser / Client
      |
  FastAPI: POST /query
      |
  Retrieval Engine
   ├── Vector Search (Qdrant)
   ├── BM25 Keyword Search
   ├── Hybrid: Custom RRF Fusion
   └── Rerank: Cross-Encoder
      |
  Guardrails
   ├── Refuse unsupported questions
   ├── Treat retrieved text as untrusted data
   └── Detect and resolve temporal conflicts
      |
  Answer + Citations + Conflict Notes
      |
  Observability: latency, token counts, errors
```
Request flow: query → selected retrieval strategy → guardrails → answer with sources → request metrics.
2. Retrieval strategies
Tier	Strategy	Purpose
1	`vector`	Finds semantically similar passages
2	`hybrid`	Combines BM25 keyword results and vector results using RRF
3	`rerank`	Reorders hybrid results with a cross-encoder
RRF combines results by rank, allowing keyword and vector retrieval to work together without comparing their different raw score scales.
3. Technology stack
Component	Technology
API / UI	FastAPI, Uvicorn, HTML
Vector database	Qdrant
Embeddings	`BAAI/bge-small-en-v1.5` (384 dimensions)
Keyword retrieval	`rank_bm25`
Fusion / reranking	Custom RRF, `BAAI/bge-reranker-base`
Answer generation	Local Ollama when configured; extractive fallback
Testing / CI	Pytest, GitHub Actions
Observability	Custom middleware and `/metrics`
Offline test settings use deterministic fake backends, so tests can run without downloading or calling live models.
4. Setup and reproduction
Docker
```bash
# Start the API and services
docker compose up --build api

# Run tests in a second terminal
docker compose --profile test run --rm tests

# Run the benchmark
docker compose --profile benchmark run --rm benchmark
```
Local Python
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000/.
Run the tests in Windows PowerShell:
```powershell
$env:EMBED_BACKEND="fake"
$env:RERANK_BACKEND="fake"
$env:LLM_BACKEND="extractive"
python -m pytest -q
```
Run the evaluation and robustness checks:
```bash
python -m app.eval.benchmark --split eval
python -m app.eval.llm_robustness
```
Example API request:
```bash
curl -s http://localhost:8000/query \
  -H "content-type: application/json" \
  -d '{"query":"how often must passwords be rotated?","strategy":"rerank"}'
```
Optional local LLM: install Ollama, run `ollama pull llama3.2:3b`, and start the app with `LLM_BACKEND=auto`.
5. Benchmark results
Recorded local-model results on the evaluation split. Raw data: `results/benchmark_eval.json`.
Strategy	Precision@5	Recall@5	MRR	NDCG@5	p50 latency	p95 latency	Peak RSS
Vector	0.2167	1.000	0.9167	0.9318	19.23 ms	29.75 ms	556.3 MB
Hybrid	0.2167	1.000	0.9167	0.9385	16.42 ms	22.46 ms	538.3 MB
Rerank	0.2167	1.000	0.9583	0.9692	872.96 ms	952.24 ms	1151.1 MB
Key findings
Rerank had the highest recorded MRR and NDCG@5, but much higher latency and memory use.
Hybrid slightly improved NDCG@5 over vector-only retrieval with low latency.
Precision@5 and Recall@5 were the same for all three methods in this run.
Evaluation used 14 queries (12 with relevance labels and 2 unsupported); the development set contains 5 queries.
Index build time was approximately 13.87 seconds in this environment.
These results are preliminary because the corpus and evaluation set are small and synthetic. Performance depends on hardware and model-cache state.
6. Cost and observability
Paid API cost: `$0` for the local setup.
Compute estimate: calculated from latency using an assumed CPU VM rate of `$0.10/hour`; this is an estimate, not an actual cloud bill.
Token usage and latency: returned by `/query` and recorded by middleware.
Error and request metrics: available at `/metrics`.
7. Robustness
Evidence checks: refuses unsupported questions when retrieved evidence is insufficient.
Prompt-injection defence: detects sample malicious snippets and treats retrieved content as untrusted data.
Temporal conflicts: highlights conflicting clauses and prioritizes the newest clause in the included retention-policy example.
Hard negatives: evaluation data includes clause IDs, numerical constraints, opposite-meaning matches, unsupported questions, and a temporal conflict.
Embedding versioning: chunks are stamped with the configured embedding version during index build.
8. Tests
The recorded Docker test run passed 59 tests: 36 unit and 23 integration.
```bash
docker compose --profile test run --rm tests
```
Coverage includes RRF and ranking, metrics, BM25, guardrails, API validation, empty results, malformed inputs, rate-limit handling, embedding-version handling, and dependency failures.
9. Known limitations
Small synthetic corpus and evaluation set; results are preliminary.
Local-model benchmark timings are hardware-dependent; CI uses deterministic fake backends.
The default `bge-reranker-base` is lighter on CPU than `bge-reranker-large`. Set `RERANK_MODEL=BAAI/bge-reranker-large` to use the larger model.
The corpus is pre-chunked JSON; PDF/DOCX/PPTX parsing is outside Option B's scope.
Evidence support uses a lexical heuristic; production use would benefit from a calibrated semantic relevance threshold.
10. Repository layout
```text
app/
  config.py
  schema.py
  embeddings.py
  ingest.py
  retrieval/       vector, BM25, RRF, reranking, engine
  eval/            metrics, benchmark, cost, LLM robustness
  llm/             guardrails and answer client
  obs/              observability middleware
  main.py
  static/index.html
data/
  corpus/corpus.json
  queries/dev_queries.json
  queries/eval_queries.json
results/
  benchmark_eval.json
  llm_robustness.txt
tests/
  unit/
  integration/
docker/
docker-compose.yml
.github/workflows/ci.yml
requirements.txt
```
