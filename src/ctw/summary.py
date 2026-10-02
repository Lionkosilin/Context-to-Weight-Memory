"""Aggregate runs that differ only in seed."""

from __future__ import annotations

import copy
import math


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return [0.0 if successes == 0 else max(0.0, center - half),
            1.0 if successes == n else min(1.0, center + half)]


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _without_seed(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c.pop("seed", None)
    c.pop("output", None)
    c.get("writer", {}).pop("save", None)
    return c


def _paired(reports, split, left, right) -> dict:
    counts = {"both": 0, "left_only": 0, "right_only": 0, "neither": 0}
    for r in reports:
        ev = r["evaluations"][split]
        if left not in ev or right not in ev:
            return {}
        rights = {(x["episode"], x["question"]): x["contains"] for x in ev[right]["rows"]}
        for row in ev[left]["rows"]:
            key = (row["episode"], row["question"])
            if key not in rights:
                continue
            lh, rh = row["contains"], rights[key]
            counts["both" if lh and rh else "left_only" if lh else "right_only" if rh else "neither"] += 1
    return {"left": left, "right": right, **counts}


def summarize(reports: list[dict]) -> dict:
    if len(reports) < 2:
        raise ValueError("need at least two runs")
    seeds = [r["config"]["seed"] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("runs must use distinct seeds")
    ref = _without_seed(reports[0]["config"])
    for r in reports[1:]:
        if _without_seed(r["config"]) != ref:
            raise ValueError("runs differ in more than the seed")
    out = {
        "note": "Wilson intervals treat items as independent; documents and seeds are clustered.",
        "seeds": seeds,
        "config": ref,
        "splits": {},
    }
    for split in reports[0]["evaluations"]:
        arms = {}
        for arm in reports[0]["evaluations"][split]:
            rows = [r["evaluations"][split][arm] for r in reports]
            hits, n = sum(x["contains"] for x in rows), sum(x["n"] for x in rows)
            rates = [x["contains"] / x["n"] for x in rows]
            arms[arm] = {
                "contains": hits, "exact": sum(x["exact"] for x in rows), "n": n,
                "rate": hits / n, "wilson95": wilson(hits, n),
                "per_seed": [x["contains"] for x in rows],
                "seed_rate_sd": _sd(rates),
                "wrong_contains": sum(x["wrong_contains"] for x in rows),
                "mean_gold_logp": sum(x["mean_gold_logp"] * x["n"] for x in rows) / n,
                **({"mean_log10_selectivity": sum(x["mean_log10_selectivity"] * x["n"] for x in rows) / n}
                   if all("mean_log10_selectivity" in x for x in rows) else {}),
            }
        out["splits"][split] = {
            "arms": arms,
            "paired": [p for p in (_paired(reports, split, "write", other)
                                   for other in ("off", "wrong", "random", "random_keys",
                                                 "random_values")) if p],
        }
    return out
