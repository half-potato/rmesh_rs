# Bench scene performance

Measured on an Apple M1 Pro (14 GPU cores, 16 GB unified memory), using the desktop viewer's High preset and a release build with the Metal backend. The Bench scene contains 650,000 vertices and 4,207,572 tetrahedra.

| Resolution | Rendering (FPS) | Frame time (ms) | Full relighting (ms) | Cached intensity update (ms) |
|---|---:|---:|---:|---:|
| 1280×720 | 18.84 | 53.08 | 1343.54 | 4.16 |
| 1920×1080 | 17.48 | 57.21 | 1345.29 | 4.30 |

Values are means over 300 frames and 10 updates per lighting operation, following warm-up. Frame times measure synchronized end-to-end rendering with vsync disabled. Relighting times measure GPU-completed updates for one point light.

Raw measurements: [720p](720p.csv), [1080p](1080p.csv).
