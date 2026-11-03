#!/usr/bin/env python3
"""Measure full-image High vs Reference MSE from matched viewer captures."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_linear_renders import compare


def summarize(directory):
    results = []
    for label, width, height in [("720p", 1280, 720), ("1080p", 1920, 1080)]:
        paths = [directory / f"{preset}-{label}.npy" for preset in ("high", "reference")]
        high, reference = [json.loads(Path(str(path) + ".json").read_text()) for path in paths]
        keys = ["width", "height", "intrinsics", "camera_position", "c2w_columns", "near", "far",
                "light_position_direction_rgb", "light_intensities", "bake"]
        for key in keys:
            if high[key] != reference[key]:
                raise ValueError(f"{label}: mismatched {key}")
        if (high["width"], high["height"]) != (width, height):
            raise ValueError(f"{label}: unexpected capture dimensions")
        for key in ("shell_enabled", "importance_enabled"):
            if high[key] is not True or reference[key] is not False:
                raise ValueError(f"{label}: unexpected {key} settings")
        report = compare(np.load(paths[1], allow_pickle=False), np.load(paths[0], allow_pickle=False),
                         reference_layout="HWC")
        if report["shape_hwc"] != [height, width, 3]:
            raise ValueError(f"{label}: image dimensions differ from metadata")
        report.update(resolution=label, candidate=str(paths[0]), reference=str(paths[1]),
                      scope="High vs Reference preset; all pixels; linear RGB")
        results.append(report)
    (directory / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    lines = ["| Resolution | Linear RGB MSE | RMSE | Mean absolute error | Maximum absolute error |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['shape_hwc'][1]}×{r['shape_hwc'][0]} | {r['mse']:.8g} | "
                     f"{r['rmse']:.8g} | {r['mean_abs_error']:.8g} | {r['max_abs_error']:.8g} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(summarize(parser.parse_args().directory))
