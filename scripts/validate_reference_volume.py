#!/usr/bin/env python3
"""Compare native checkpoint lighting with a matched sdir linear RGB image.

First run sdir_checkpoint_reference.py with --export-scene, then pass that scene,
its output image, and the same camera JSON here. The reference script writes the
exact rays beside its image. Requires numpy and a built rmesh-viewer binary.
This validates the experimental command-line reference path, not the viewer's
vertex-average bake. No exposure adjustment, masking, or tone mapping is used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from compare_linear_renders import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('target/release/rmesh-viewer'))
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--camera-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=.01)
    args = parser.parse_args()
    camera = json.loads(args.camera_json.read_text())
    reference = np.load(args.reference, allow_pickle=False)
    if reference.ndim != 3 or reference.shape[0] != 3:
        raise ValueError('Expected CHW reference RGB')
    h, w = reference.shape[1:]
    rays = Path(str(args.reference) + '.rays.bin')
    if rays.stat().st_size != h * w * 24:
        raise ValueError('Ray count does not match reference image')
    result = np.zeros((h, w, 3), np.float32)
    lights = camera['light_position_direction_rgb']
    intensities = camera['light_intensities']
    if len(lights) != len(intensities):
        raise ValueError('Light and intensity counts differ')
    with tempfile.TemporaryDirectory(prefix='rmesh-validation-') as directory:
        raw = Path(directory) / 'output.rgba'
        for light, intensity in zip(lights, intensities):
            position, _, rgb = light
            if intensity <= 0:
                continue
            subprocess.run([str(args.binary.resolve()), '--reference-volume',
                            str(args.scene), str(rays), str(raw),
                            *map(str, position), str(intensity), str(args.samples),
                            str(camera['near']), *map(str, rgb)], check=True)
            result += np.fromfile(raw, dtype='<f4').reshape(h, w, 4)[..., :3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    report = compare(reference, result, epsilon=args.epsilon)
    report['reference'] = str(args.reference)
    report['scene'] = str(args.scene)
    report['scope'] = 'native per-sample reference path; CPU-evaluated checkpoint; matched point lights'
    text = json.dumps(report, indent=2, allow_nan=False)
    Path(str(args.output) + '.json').write_text(text + '\n')
    print(text)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
