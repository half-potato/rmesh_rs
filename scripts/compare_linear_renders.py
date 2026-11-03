#!/usr/bin/env python3
"""Gate matched, pre-tone-map linear RGB renders against an absolute epsilon.

Export sdir's result['volumetric_linear'] as a float .npy (CHW), not pbr_rgb.
Camera rays, dimensions, lights, geometry/materials, quadrature settings and
background must match. No alignment, masking, exposure fitting, clipping or
gamma conversion is performed here: those would hide renderer differences.

Capture the current viewer (unverified approximation) with:
  RMESH_QUALITY=Reference RMESH_WIN=320x180 RMESH_SHOT=/tmp/candidate.png \
      RMESH_LINEAR_SHOT=/tmp/candidate.npy rmesh-viewer /path/to/bench.rmesh
The .npy.json sidecar records camera/light settings. Reference quality disables
shell/importance culling; linear capture also hides opaque helpers/light gizmos.
Float32 export preserves the attachment's values; it cannot recover precision
already lost in a float16 render target. A passing image pair certifies only
that configuration, not all camera/light positions or unexported material data.

Example:
  python3 scripts/compare_linear_renders.py reference.npy candidate.npy \
      --reference-layout CHW --candidate-layout HWC --epsilon 0.01
Exit 0: every RGB channel passes; 1: mismatch; 2: invalid input.
"""
import argparse
import json
import sys

import numpy as np


def rgb_image(image, layout):
    image = np.asarray(image)
    if image.dtype.kind != "f":
        raise ValueError("Expected floating-point linear RGB, not integer/display pixels")
    if image.ndim != 3:
        raise ValueError("Expected a three-dimensional RGB image")
    if layout == "CHW":
        image = np.moveaxis(image, 0, -1)
    elif layout != "HWC":
        raise ValueError("Layout must be HWC or CHW")
    if image.shape[-1] != 3 or image.size == 0:
        raise ValueError("Expected exactly three RGB channels and nonempty dimensions")
    if not np.isfinite(image).all():
        raise ValueError("NaN/Inf in render: traversal or shading failure")
    return image.astype(np.float64)


def compare(reference, candidate, epsilon=1e-2, reference_layout="CHW", candidate_layout="HWC"):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    reference = rgb_image(reference, reference_layout)
    candidate = rgb_image(candidate, candidate_layout)
    if reference.shape != candidate.shape:
        raise ValueError(f"Image dimensions differ: {reference.shape} versus {candidate.shape}")
    error = np.abs(candidate - reference)
    worst = tuple(int(x) for x in np.unravel_index(np.argmax(error), error.shape))
    failed = np.any(error > epsilon, axis=-1)
    return {
        "passed": bool(not failed.any()),
        "metric": "maximum_absolute_linear_rgb_error",
        "epsilon": epsilon,
        "shape_hwc": list(reference.shape),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "mse": float(np.mean(error * error)),
        "mse_rgb": np.mean(error * error, axis=(0, 1)).tolist(),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs_error_rgb": error.max(axis=(0, 1)).tolist(),
        "failed_pixels": int(failed.sum()),
        "total_pixels": int(failed.size),
        "worst_y_x_channel": list(worst),
        "reference_at_worst": float(reference[worst]),
        "candidate_at_worst": float(candidate[worst]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--reference-layout", choices=["CHW", "HWC"], default="CHW")
    parser.add_argument("--candidate-layout", choices=["CHW", "HWC"], default="HWC")
    args = parser.parse_args()
    try:
        report = compare(
            np.load(args.reference, allow_pickle=False),
            np.load(args.candidate, allow_pickle=False),
            args.epsilon, args.reference_layout, args.candidate_layout,
        )
    except (OSError, ValueError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
