
"""
Benchmark runner for Option B: Advanced Hybrid Retrieval & Reranking.

Compares:
1. Vector-only retrieval
2. Hybrid BM25 + vector retrieval
3. Hybrid retrieval + cross-encoder reranking

Reports Precision@k, Recall@k, MRR, NDCG@5, p50/p95 latency,
CPU utilization, peak RSS memory, and estimated compute cost.

Run:
    python -m app.eval.benchmark --split eval
    python -m app.eval.benchmark --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import psutil

from app.config import settings
from app.eval import cost, metrics
from app.retrieval.engine import RetrievalEngine

STRATEGIES = ["vector", "hybrid", "rerank"]


def _load_queries(split: str) -> List[dict]:
    path = (
        settings.eval_queries_path
        if split == "eval"
        else settings.dev_queries_path
    )
    queries = json.loads(Path(path).read_text(encoding="utf-8"))

    if not queries:
        raise ValueError(f"No queries found in {path}")

    required_fields = {"query", "relevant_chunk_ids"}
    for index, query in enumerate(queries):
        missing = required_fields - query.keys()
        if missing:
            raise ValueError(
                f"Query {index} in {path} is missing: {sorted(missing)}"
            )

    return queries


def _measure_cpu_percent(
    proc: psutil.Process,
    operation,
) -> tuple[float, object]:
    """Measure process CPU utilization while an operation runs."""
    proc.cpu_percent(None)
    start = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - start
    cpu_percent = proc.cpu_percent(None)

    return round(cpu_percent, 1), (result, elapsed)


def run_benchmark(split: str = "eval") -> dict:
    queries = _load_queries(split)

    # Includes engine initialization, embedding/index setup and related work.
    build_start = time.perf_counter()
    engine = RetrievalEngine()
    index_build_ms = round(
        (time.perf_counter() - build_start) * 1000.0, 2
    )

    k = settings.top_k
    proc = psutil.Process(os.getpid())
    matrix: dict[str, dict] = {}

    # Unsupported queries are evaluated separately for refusal robustness.
    # Do not let them artificially inflate retrieval relevance metrics.
    relevance_queries = [
        query
        for query in queries
        if query["relevant_chunk_ids"]
    ]

    if not relevance_queries:
        raise ValueError(
            f"The {split} split contains no queries with relevant chunk IDs."
        )

    for strategy in STRATEGIES:
        # Warm up before measuring latency.
        engine.retrieve(queries[0]["query"], strategy=strategy, top_k=k)

        p_scores = []
        r_scores = []
        mrr_scores = []
        ndcg_scores = []
        latencies: List[float] = []

        # RSS is process-wide, not strategy-exclusive memory.
        peak_rss = proc.memory_info().rss
        cpu_samples = []

        for query in queries:
            # Measure CPU over each retrieval operation. Very short operations
            # can produce noisy CPU samples, so treat this as approximate.
            cpu, (retrieved, elapsed) = _measure_cpu_percent(
                proc,
                lambda q=query: engine.retrieve(
                    q["query"], strategy=strategy, top_k=k
                ),
            )

            latencies.append(elapsed * 1000.0)
            cpu_samples.append(cpu)
            peak_rss = max(peak_rss, proc.memory_info().rss)

            # Unsupported queries are excluded from relevance averages.
            if not query["relevant_chunk_ids"]:
                continue

            retrieved_ids = [chunk.chunk_id for chunk in retrieved]
            gold = query["relevant_chunk_ids"]

            p_scores.append(metrics.precision_at_k(retrieved_ids, gold, k))
            r_scores.append(metrics.recall_at_k(retrieved_ids, gold, k))
            mrr_scores.append(metrics.mrr(retrieved_ids, gold))
            ndcg_scores.append(metrics.ndcg_at_k(retrieved_ids, gold, 5))

        p50 = metrics.percentile(latencies, 50)
        p95 = metrics.percentile(latencies, 95)

        def average(values):
            return round(sum(values) / len(values), 4) if values else 0.0

        avg_latency_ms = (
            sum(latencies) / len(latencies) if latencies else 0.0
        )

        matrix[strategy] = {
            "precision_at_k": average(p_scores),
            "recall_at_k": average(r_scores),
            "mrr": average(mrr_scores),
            "ndcg_at_5": average(ndcg_scores),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "avg_cpu_percent": round(average(cpu_samples), 1),
            "peak_rss_mb": round(peak_rss / (1024 * 1024), 1),
            "avg_compute_cost_usd": round(
                cost.compute_cost_usd(avg_latency_ms), 8
            ),
            "k": k,
            "num_queries": len(queries),
            "num_relevance_queries": len(relevance_queries),
            "num_unsupported_queries": (
                len(queries) - len(relevance_queries)
            ),
        }

    report = {
        "split": split,
        "embedding_version": settings.embed_version,
        "index_build_ms": index_build_ms,
        "corpus_size": len(engine.chunks),
        "total_queries": len(queries),
        "relevance_queries": len(relevance_queries),
        "unsupported_queries": len(queries) - len(relevance_queries),
        "metrics_note": (
            "Retrieval relevance metrics exclude queries with empty "
            "relevant_chunk_ids. Evaluate those queries separately for "
            "LLM refusal robustness. CPU and RSS are approximate "
            "process-level measurements; compute cost is an estimate."
        ),
        "matrix": matrix,
    }

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"benchmark_{split}.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    _print_table(report)
    print(f"\nIndex build time: {index_build_ms} ms")
    print(f"Raw results written to: {output_path}")

    return report


def _print_table(report: dict) -> None:
    columns = [
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_5",
        "p50_latency_ms",
        "p95_latency_ms",
        "avg_cpu_percent",
        "peak_rss_mb",
        "avg_compute_cost_usd",
    ]

    print(
        f"\nBenchmark: {report['split']} split | "
        f"queries={report['total_queries']} | "
        f"relevance queries={report['relevance_queries']} | "
        f"corpus={report['corpus_size']}"
    )

    header = ["strategy"] + columns
    print(" | ".join(f"{name:>20}" for name in header))
    print("-" * (23 * len(header)))

    for strategy, row in report["matrix"].items():
        values = [strategy] + [str(row[column]) for column in columns]
        print(" | ".join(f"{value:>20}" for value in values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the retrieval benchmark."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "eval"],
        default="eval",
    )
    args = parser.parse_args()
    run_benchmark(args.split)
