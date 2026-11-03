#!/usr/bin/env python3
"""Capture a lit mesh repeatedly with the shell enabled and reject black/flickering frames.

Example: python3 scripts/check_shell_stability.py ~/Downloads/bench.rmesh
Requires numpy and a release viewer build. Uses an actual window; reports the
actual render size because desktop window managers may cap the requested size.
The synthetic shell_stability_test separately exercises a true offscreen 4K target.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/release/rmesh-viewer'))
    parser.add_argument('--size', default='1920x1080')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--epsilon', type=float, default=.001)
    args = parser.parse_args()
    if args.frames < 2 or not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error('Need at least two frames and a finite positive epsilon')
    with tempfile.TemporaryDirectory(prefix='shell-stability-') as directory:
        prefix = Path(directory) / 'frame'
        env = dict(os.environ)
        # GPU instrumentation can hide the original Metal failure.
        for name in ('MTL_SHADER_VALIDATION', 'MTL_DEBUG_LAYER', 'RMESH_HOT_SHADERS'):
            env.pop(name, None)
        env.update(RMESH_QUALITY='High', RMESH_SHELL='1', RMESH_WIN=args.size,
                   RMESH_SHOT=str(prefix) + '.png', RMESH_LINEAR_SHOT=str(prefix),
                   RMESH_SHOT_COUNT=str(args.frames), RMESH_SHOT_FRAME='100')
        subprocess.run([str(args.binary.resolve()), str(args.mesh.expanduser().resolve())],
                       env=env, check=True, timeout=300)
        first = np.load(str(prefix) + '.0.npy', allow_pickle=False)
        peak_error = 0.
        for index in range(args.frames):
            path = str(prefix) + f'.{index}.npy'
            frame = np.load(path, allow_pickle=False)
            metadata = json.loads(Path(path + '.json').read_text())
            if not metadata['shell_enabled']:
                raise AssertionError('Shell must be enabled during this regression')
            if frame.shape != first.shape or not np.isfinite(frame).all() or frame.max() <= .01:
                raise AssertionError(f'Frame {index}: invalid or dark capture; use a visibly lit mesh')
            error = float(np.max(np.abs(frame - first)))
            peak_error = max(peak_error, error)
            if error > args.epsilon:
                raise AssertionError(f'Frame {index}: RGB variation {error} exceeds {args.epsilon}')
        print(json.dumps(dict(passed=True, frames=args.frames,
                              actual_shape_hwc=list(first.shape), max_rgb_variation=peak_error)))


if __name__ == '__main__':
    main()
