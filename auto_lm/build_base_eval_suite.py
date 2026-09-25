#!/usr/bin/env python3
"""Build the immutable sixteen-candidate Base evaluation suite from official MMPO assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


OFFICIAL_SPECS = {
    "MATH500": ("MATH500/math500.jsonl", 500),
    "OlymMATH": ("Olympiad-math/olympiad.jsonl", 675),
    "AMC23": ("AMC23/amc23.jsonl", 40),
    "AIME24": ("AIME2024/aime2024.jsonl", 30),
    "AIME25": ("AIME2025/aime2025.jsonl", 30),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_suite(
    eval_root: Path,
    output_dir: Path,
    source_commit: str,
    candidates_per_problem: int = 16,
    specs: Mapping[str, tuple[str, int]] = OFFICIAL_SPECS,
) -> dict:
    if candidates_per_problem != 16:
        raise ValueError("the controlled Base protocol requires exactly 16 candidates")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output directory must be absent or empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "base_eval.jsonl"
    problems: dict[str, dict] = {}
    source_assets: dict[str, dict] = {}
    combined_rows: list[dict] = []

    for benchmark, (relative_path, expected_count) in specs.items():
        source_path = eval_root / relative_path
        rows = load_jsonl(source_path)
        if len(rows) != expected_count:
            raise ValueError(
                f"unexpected source count for {benchmark}: expected={expected_count} observed={len(rows)}"
            )
        source_assets[benchmark] = {
            "path": str(source_path),
            "rows": len(rows),
            "sha256": sha256(source_path),
        }
        for position, source_row in enumerate(rows):
            row = dict(source_row)
            extra_info = dict(row.get("extra_info") or {})
            problem_id = extra_info.get("index", position)
            identity = f"table2::{benchmark}::{problem_id}"
            if identity in problems:
                raise ValueError(f"duplicate evaluation identity: {identity}")
            extra_info.update(
                {
                    "table2_id": identity,
                    "benchmark": benchmark,
                    "problem_id": problem_id,
                }
            )
            row["extra_info"] = extra_info
            combined_rows.append(row)
            problems[identity] = {
                "benchmark": benchmark,
                "problem_id": problem_id,
                "source_position": position,
            }

    with combined_path.open("w", encoding="utf-8") as handle:
        for row in combined_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "suite": "mmpo_table2_base",
        "source_commit": source_commit,
        "candidates_per_problem": candidates_per_problem,
        "problem_count": len(problems),
        "expected_generations": len(problems) * candidates_per_problem,
        "dataset": {
            "path": str(combined_path),
            "rows": len(combined_rows),
            "sha256": sha256(combined_path),
        },
        "source_assets": source_assets,
        "problems": problems,
    }
    manifest_path = output_dir / "base_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--candidates-per-problem", type=int, default=16)
    args = parser.parse_args()
    manifest = build_suite(
        eval_root=args.eval_root,
        output_dir=args.output_dir,
        source_commit=args.source_commit,
        candidates_per_problem=args.candidates_per_problem,
    )
    print(
        "BASE_EVAL_SUITE_PASS "
        f"problems={manifest['problem_count']} generations={manifest['expected_generations']}"
    )


if __name__ == "__main__":
    main()
