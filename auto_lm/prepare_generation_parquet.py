#!/usr/bin/env python3
"""Convert the controlled JSONL suite to the parquet schema used by MMPO generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("data_source", "prompt", "reward_model", "extra_info")


def _load_jsonl(source: Path) -> list[dict]:
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {source}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"suite row {line_number} is not an object")
            missing = [column for column in REQUIRED_COLUMNS if column not in row]
            if missing:
                raise ValueError(f"suite row {line_number} is missing columns: {missing}")
            extra_info = row["extra_info"]
            if not isinstance(extra_info, dict):
                raise ValueError(f"suite row {line_number} extra_info is not an object")
            for key in ("table2_id", "benchmark", "problem_id"):
                if key not in extra_info:
                    raise ValueError(f"suite row {line_number} extra_info is missing {key}")
            rows.append({column: row[column] for column in REQUIRED_COLUMNS})
    if not rows:
        raise ValueError("the generation suite is empty")
    return rows


def prepare(source: Path, output: Path) -> dict:
    """Write an identity-preserving parquet input and return its audit metadata."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = _load_jsonl(source)
    frame = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {
        "schema_version": 1,
        "rows": len(frame),
        "columns": list(frame.columns),
        "source": str(source),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    metadata = prepare(args.source, args.output)
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"BASE_GENERATION_PARQUET_PASS rows={metadata['rows']} sha256={metadata['sha256']}")


if __name__ == "__main__":
    main()
