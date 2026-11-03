#!/usr/bin/env python3
"""Validate viewer benchmark CSV files and print a Markdown results table."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(paths):
    rows = []
    for path in paths:
        groups = defaultdict(list)
        dimensions = set()
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                ms = float(row["ms"])
                if not math.isfinite(ms) or ms <= 0:
                    raise ValueError(f"{path}: invalid duration {ms}")
                dimensions.add((int(row["width"]), int(row["height"])))
                groups[row["operation"]].append(ms)
        if len(dimensions) != 1:
            raise ValueError(f"{path}: resolution changed during benchmark")
        w, h = dimensions.pop()
        for operation, count in [("render", 300), ("relight_full", 10), ("relight_cached", 10)]:
            values = groups[operation]
            if len(values) != count:
                raise ValueError(f"{path}: expected {count} {operation} samples, got {len(values)}")
            mean = statistics.mean(values)
            p95 = sorted(values)[math.ceil(len(values) * 0.95) - 1]
            rows.append(f"| {w}×{h} | {operation} | {len(values)} | {mean:.2f} | "
                        f"{statistics.stdev(values):.2f} | {statistics.median(values):.2f} | "
                        f"{p95:.2f} | {1000 / mean:.2f} |")
    return "\n".join([
        "| Resolution | Operation | N | Mean (ms) | SD (ms) | Median (ms) | P95 (ms) | Rate (s⁻¹) |",
        "|---|---|---:|---:|---:|---:|---:|---:|", *rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    print(summarize(parser.parse_args().csv))
