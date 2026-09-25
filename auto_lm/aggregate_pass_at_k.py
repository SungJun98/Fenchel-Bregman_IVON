#!/usr/bin/env python3
"""Fail-closed Pass@K aggregation for the controlled MMPO Base evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


BENCHMARKS = ("MATH500", "OlymMATH", "AMC23", "AIME24", "AIME25")
PASS_KS = (1, 4, 8, 16)
SINGLE_SAMPLE_BENCHMARKS = {"MATH500", "OlymMATH"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pass_at_k(total: int, correct: int, k: int) -> float:
    if not 0 <= correct <= total:
        raise ValueError("correct candidate count is outside the valid range")
    if not 1 <= k <= total:
        raise ValueError("K is outside the valid range")
    if total - correct < k:
        return 1.0
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def aggregate(manifest: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
    if manifest.get("suite") != "mmpo_table2_base":
        raise ValueError("expected the controlled MMPO Base manifest")
    candidate_count = manifest.get("candidates_per_problem")
    if candidate_count != 16:
        raise ValueError("expected sixteen candidates per problem")
    expected = manifest.get("problems", {})
    if len(expected) != manifest.get("problem_count"):
        raise ValueError("manifest problem count mismatch")
    if len(rows) != manifest.get("expected_generations"):
        raise ValueError(
            "candidate count mismatch: "
            f"expected={manifest.get('expected_generations')} observed={len(rows)}"
        )

    observed: dict[str, list[dict]] = defaultdict(list)
    candidates: list[dict] = []
    for row in rows:
        identity = row.get("table2_id")
        if identity not in expected:
            raise ValueError(f"unknown evaluation identity: {identity}")
        benchmark = row.get("benchmark")
        if benchmark != expected[identity]["benchmark"]:
            raise ValueError(f"benchmark mismatch for {identity}")
        score = row.get("score")
        if score not in (0, 0.0, 1, 1.0):
            raise ValueError(f"nonbinary evaluation score for {identity}: {score}")
        candidate_index = len(observed[identity])
        provided_index = row.get("candidate_index")
        if provided_index is not None:
            try:
                normalized_index = int(provided_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"candidate index mismatch for {identity}: {provided_index}") from exc
            if normalized_index != candidate_index:
                raise ValueError(
                    f"candidate index mismatch for {identity}: "
                    f"expected={candidate_index} observed={provided_index}"
                )
        annotated = dict(row)
        annotated["candidate_index"] = candidate_index
        annotated["score"] = float(score)
        observed[identity].append(annotated)
        candidates.append(annotated)

    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(f"incomplete evaluation coverage: missing={missing} extra={extra}")
    for identity, values in observed.items():
        if len(values) != candidate_count:
            raise ValueError(
                f"candidate count mismatch for {identity}: expected={candidate_count} observed={len(values)}"
            )

    per_benchmark_problem_metrics: dict[str, list[dict]] = defaultdict(list)
    paper_scores: dict[str, list[float]] = defaultdict(list)
    for identity, definition in expected.items():
        values = observed[identity]
        scores = [value["score"] for value in values]
        correct = int(sum(scores))
        benchmark = definition["benchmark"]
        per_benchmark_problem_metrics[benchmark].append(
            {
                "table2_id": identity,
                "correct_candidates": correct,
                **{f"pass@{k}": pass_at_k(candidate_count, correct, k) for k in PASS_KS},
            }
        )
        if benchmark in SINGLE_SAMPLE_BENCHMARKS:
            paper_scores[benchmark].append(scores[0])
        else:
            paper_scores[benchmark].append(sum(scores) / candidate_count)

    if set(per_benchmark_problem_metrics) != set(BENCHMARKS):
        raise ValueError("incomplete benchmark coverage")

    benchmark_metrics = {}
    for benchmark in BENCHMARKS:
        problem_values = per_benchmark_problem_metrics[benchmark]
        benchmark_metrics[benchmark] = {
            "problem_count": len(problem_values),
            "correct_candidates": sum(value["correct_candidates"] for value in problem_values),
            "candidate_accuracy": sum(value["correct_candidates"] for value in problem_values)
            / (len(problem_values) * candidate_count),
            **{
                f"pass@{k}": sum(value[f"pass@{k}"] for value in problem_values) / len(problem_values)
                for k in PASS_KS
            },
        }

    pass_at_k_avg = {
        f"pass@{k}": sum(benchmark_metrics[name][f"pass@{k}"] for name in BENCHMARKS) / len(BENCHMARKS)
        for k in PASS_KS
    }
    paper_benchmark_scores = {
        benchmark: sum(values) / len(values) for benchmark, values in paper_scores.items()
    }
    result = {
        "schema_version": 1,
        "suite": manifest["suite"],
        "source_commit": manifest.get("source_commit"),
        "candidates_per_problem": candidate_count,
        "problem_count": len(expected),
        "generation_count": len(rows),
        "benchmark_metrics": benchmark_metrics,
        "pass_at_k_avg": pass_at_k_avg,
        "primary_metric": {"name": "Pass@8 Avg.", "value": pass_at_k_avg["pass@8"]},
        "paper_compatible": {
            "definition": "candidate 0 for MATH500/OlymMATH; mean of 16 for AMC23/AIME24/AIME25",
            "benchmark_scores": paper_benchmark_scores,
            "avg": sum(paper_benchmark_scores[name] for name in BENCHMARKS) / len(BENCHMARKS),
        },
        "problem_metrics": {
            benchmark: per_benchmark_problem_metrics[benchmark] for benchmark in BENCHMARKS
        },
    }
    return result, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for path in args.generations:
        rows.extend(load_jsonl(path))
    result, candidates = aggregate(manifest, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates_output.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        "BASE_EVALUATION_PASS "
        f"problems={result['problem_count']} generations={result['generation_count']} "
        f"pass_at_8_avg={result['primary_metric']['value']:.10f}"
    )


if __name__ == "__main__":
    main()
