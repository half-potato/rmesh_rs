# rmeshvk Wiki

Technical documentation for the rmeshvk renderer, written in LaTeX and
compiled to a single PDF — **[rmeshvk-wiki.pdf](rmeshvk-wiki.pdf)** — with
[tectonic](https://tectonic-typesetting.github.io/). Rebuild with
`./build.sh`. The master document is `rmeshvk-wiki.tex`; each chapter lives
in its own `.tex` file:

| Chapter | Contents |
|---|---|
| `overview.tex` | Project architecture: radiance meshes, the circumsphere power sort, the volume rendering integral, workspace crates, the three rendering pathways, viewer frame graph, `.rmesh` format |
| `interval-renderer.tex` | The main renderer: projection & SH color, interval decomposition of tets into screen-space triangles, fixed-slot buffers, the fragment integral, MRT outputs, mesh-shader variant |
| `mesh-style-renderers.tex` | Legacy renderers: 12-triangle forward, quad billboard, mesh-shader forward, and the ray-trace diagnostic mode |
| `tile-renderer.tex` | The trainable pathway: compute-based tiled forward, reverse-order backward with state undo, warp-reduced gradient accumulation, chain rule back to mesh parameters |
| `deferred-shading.tex` | G-buffer layout, GGX PBR lighting, GTAO/SSGI/SSR screen-space stack, tone mapping, debug modes, and the non-deferred compositor |
| `shadow-rendering.tex` | Moment-based deep shadow maps: 2-moment representation, per-face generation, Cantelli transmittance bound, cubemap atlas |
| `gpu-sorting.tex` | The power sort key, Basic and DRS radix sort backends, CPU fallback, and 64-bit tile-pair sorting |
| `training-and-python.tex` | GPU-only training loop (loss, Adam, error stats) and the PyTorch bridge (`RMeshRenderer`, autograd) |
