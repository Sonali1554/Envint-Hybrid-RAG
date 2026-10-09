Envint — Hybrid Retrieval & Reranking Benchmark
AI Engineer Intern Technical Assignment · Option B
A compliance question-answering app that compares vector search, hybrid BM25 + vector retrieval with Reciprocal Rank Fusion (RRF), and cross-encoder reranking. It includes source citations, prompt-injection handling, temporal conflict notes, evaluation scripts, tests, and Docker support.
> Models run locally; no paid API is required. Localhost links work only while the app is running on your computer.
Quick links
Web UI
API docs
Health check
Metrics
1. Architecture
```text
Browser / Client
      |
  FastAPI: POST /query
      |
  Retrieval engine
   ├── Vector search (Qdrant)
   ├── BM25 keyword search
   ├── Hybrid: custom RRF fusion
   └── Rerank: cross-encoder
      |
  Guardrails
   ├── Evidence checks / refusal
   ├── Prompt-injection isolation
   └── Temporal conflict handling
      |
  Answer + source clauses
      |
  Request metrics: latency, tokens, errors
```
2. Retrieval strategies
Strategy	Method	Main trade-off
`vector`	Semantic search with embeddings	Good for meaning and paraphrases
`hybrid`	BM25 + vector results combined using RRF	Balances keywords and semantic matches
`rerank`	Hybrid candidates reordered by a cross-encoder	Better ranking in this evaluation, but slower
RRF combines result ranks, rather than comparing BM25 and vector scores directly.
3. Technology stack
Component	Technology
API and UI	FastAPI, Uvicorn, HTML
Vector database	Qdrant
Embeddings	`BAAI/bge-small-en-v1.5`
Keyword retrieval / fusion	BM25 (`rank_bm25`), custom RRF
Reranker	`BAAI/bge-reranker-base` by default
Answer generation	Local Ollama when configured; extractive fallback
Tests and CI	Pytest, GitHub Actions
Observability	Custom request middleware and `/metrics`
Offline test backends use deterministic stand-ins, allowing the test suite to run without downloading or calling an LLM.
4. Run the project
Docker
```bash
# Start the API and its services
docker compose up --build api

# In another terminal: run the full test suite
docker compose --profile test run --rm tests

# Run the evaluation benchmark
docker compose --profile benchmark run --rm benchmark
```
Local Python
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000/.
Run tests in Windows PowerShell with offline backends:
```powershell
$env:EMBED_BACKEND="fake"
$env:RERANK_BACKEND="fake"
$env:LLM_BACKEND="extractive"
python -m pytest -q
```
Run the benchmark and robustness checks:
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
Optional local LLM: install Ollama, run `ollama pull llama3.2:3b`, then start the app with `LLM_BACKEND=auto`.
5. Evaluation results
Recorded results from the local model run. Raw output: `results/benchmark_eval.json`.
Strategy	Precision@5	Recall@5	MRR	NDCG@5	p50 latency	p95 latency	Peak RSS
Vector	0.2167	1.000	0.9167	0.9318	19.23 ms	29.75 ms	556.3 MB
Hybrid	0.2167	1.000	0.9167	0.9385	16.42 ms	22.46 ms	538.3 MB
Rerank	0.2167	1.000	0.9583	0.9692	872.96 ms	952.24 ms	1151.1 MB
Summary: Reranking achieved the best recorded MRR and NDCG@5, but had much higher latency and memory use. Hybrid retrieval had competitive ranking quality with much lower latency. Precision@5 and Recall@5 were identical across strategies in this run.
Evaluation set: 14 queries (12 with relevance labels and 2 unsupported); development set: 5 queries.
Recorded index build time: approximately 13.87 seconds.
These are preliminary results from a small synthetic corpus and are hardware-dependent.
6. Cost and observability
No paid API is called, so API cost is $0 for the local setup. The cost script estimates compute cost using an assumed CPU VM rate of `$0.10/hour`; this is a proxy, not a cloud bill. Hosted-model token costs are shown only as a comparison.
The API response includes token usage and latency. The `/metrics` endpoint exposes request metrics, including latency and errors.
7. Robustness and data handling
Refuses questions when retrieved evidence does not support an answer.
Treats retrieved document text as untrusted data, not instructions.
Detects prompt-injection patterns in sample documents.
Highlights temporal conflicts and selects the newest clause in the included retention-policy example.
Includes hard negatives, clause IDs, numeric constraints, unsupported questions, and conflict cases in the query data.
8. Tests
The recorded Docker test run passed 59 tests: 36 unit and 23 integration.
```bash
docker compose --profile test run --rm tests
```
Coverage includes retrieval and RRF ranking, evaluation metrics, guardrails, API validation, empty results, malformed inputs, rate-limit handling, embedding-version handling, and dependency failures.
9. Limitations
The benchmark corpus and evaluation set are small and synthetic; results are preliminary.
Benchmark timings were recorded with real local models. CI uses deterministic fake backends, so CI timings are not comparable.
The default `bge-reranker-base` is lighter on CPU than `bge-reranker-large`. Set `RERANK_MODEL=BAAI/bge-reranker-large` to try the larger model.
The corpus is pre-chunked JSON. Parsing PDF/DOCX/PPTX files is outside the scope of Option B.
Evidence support currently uses a lexical heuristic; production use would benefit from a calibrated semantic relevance threshold.
10. Repository layout
```text
app/
  config.py
  schema.py
  embeddings.py
  ingest.py
  retrieval/       vector, BM25, RRF, reranking, retrieval engine
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
