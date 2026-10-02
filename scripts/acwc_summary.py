#!/usr/bin/env python3
"""Aggregate ACWC runs across seeds, with per-seed validation and Wilson intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CONDITIONS = (
    "off",
    "full_context_oracle",
    "matched_compiled_weights",
    "wrong_document_weights",
    "random_same_rank_norm",
)
SPLITS = ("seen_recombination", "unseen_values")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == n else min(1.0, center + half)
    return [lower, upper]


def validate(reports: list[dict]) -> None:
    if len(reports) < 2:
        raise ValueError("at least two replications are required")
    seeds = [report["seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("replication seeds must be distinct")
    keys = (
        "method", "model_dir", "layer", "train_docs", "dev_docs",
        "test_docs_per_split", "steps", "learning_rate", "route_weight",
        "route_temperature", "key_residual_rank", "value_residual_rank",
        "value_source", "state_rank", "state_bytes_per_document_bf16",
    )
    reference = reports[0]
    for report in reports[1:]:
        for key in keys:
            if report.get(key) != reference.get(key):
                raise ValueError(
                    f"replication mismatch for {key}: "
                    f"{reference.get(key)!r} != {report.get(key)!r}"
                )
    for report in reports:
        for split in SPLITS:
            if set(report["evaluations"][split]) != set(CONDITIONS):
                raise ValueError(f"condition mismatch in seed {report['seed']} {split}")


def aggregate_condition(reports: list[dict], split: str, condition: str) -> dict:
    rows = [report["evaluations"][split][condition] for report in reports]
    successes = sum(row["contains"] for row in rows)
    exact = sum(row["exact"] for row in rows)
    n = sum(row["n"] for row in rows)
    rates = [row["contains"] / row["n"] for row in rows]
    weighted_logp = sum(row["mean_gold_logp"] * row["n"] for row in rows) / n
    return {
        "successes": successes,
        "exact": exact,
        "n": n,
        "rate": successes / n,
        "wilson_95_item_level": wilson(successes, n),
        "per_seed_successes": [row["contains"] for row in rows],
        "per_seed_rates": rates,
        "mean_seed_rate": mean(rates),
        "sample_sd_seed_rate": sample_sd(rates),
        "mean_gold_logp": weighted_logp,
        "wrong_value_successes": sum(row["wrong_value_contains"] for row in rows),
    }


def paired_table(reports: list[dict], split: str,
                 left: str, right: str) -> dict:
    counts = {"both": 0, "left_only": 0, "right_only": 0, "neither": 0}
    for report in reports:
        left_rows = report["evaluations"][split][left]["rows"]
        right_rows = report["evaluations"][split][right]["rows"]
        if len(left_rows) != len(right_rows):
            raise ValueError("paired conditions have different row counts")
        for lrow, rrow in zip(left_rows, right_rows):
            identity = (lrow["doc_index"], lrow["relation"], lrow["gold"])
            other = (rrow["doc_index"], rrow["relation"], rrow["gold"])
            if identity != other:
                raise ValueError("paired row identities do not align")
            lhit = bool(lrow["contains"])
            rhit = bool(rrow["contains"])
            if lhit and rhit:
                counts["both"] += 1
            elif lhit:
                counts["left_only"] += 1
            elif rhit:
                counts["right_only"] += 1
            else:
                counts["neither"] += 1
    return {"left": left, "right": right, **counts}


def main() -> None:
    args = parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    validate(reports)
    aggregate = {
        "status": "development_replication_aggregate",
        "inference_warning": (
            "Wilson intervals treat items as independent and are descriptive only; "
            "documents and seeds are clustered."
        ),
        "inputs": [str(path) for path in args.inputs],
        "seeds": [report["seed"] for report in reports],
        "n_runs": len(reports),
        "frozen_configuration": {
            key: reports[0].get(key) for key in (
                "method", "model_dir", "layer", "train_docs", "dev_docs",
                "test_docs_per_split", "steps", "learning_rate", "route_weight",
                "route_temperature", "key_residual_rank", "value_residual_rank",
                "value_source", "state_rank", "state_bytes_per_document_bf16",
                "compiler_parameter_count",
            )
        },
        "splits": {},
    }
    for split in SPLITS:
        aggregate["splits"][split] = {
            "conditions": {
                condition: aggregate_condition(reports, split, condition)
                for condition in CONDITIONS
            },
            "paired_matched_vs_off": paired_table(
                reports, split, "matched_compiled_weights", "off"
            ),
            "paired_matched_vs_wrong_document": paired_table(
                reports, split, "matched_compiled_weights", "wrong_document_weights"
            ),
            "paired_matched_vs_random": paired_table(
                reports, split, "matched_compiled_weights", "random_same_rank_norm"
            ),
        }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
