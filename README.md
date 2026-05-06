# Mapillary × MapAnything — Street-Level 3D Reconstruction Pipeline

A research pipeline for metric 3D reconstruction, trajectory evaluation, and Gaussian Splatting from Mapillary street-level imagery, powered by Meta's [MapAnything](https://github.com/facebookresearch/map-anything) feed-forward transformer.

---

## Overview

This project pulls geo-tagged image sequences from the Mapillary API and feeds them through MapAnything to produce:

- Globally consistent dense point clouds with per-pixel color and confidence
- Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) metrics against GPS ground truth
- Naive and improved `.splat` files for in-browser Gaussian Splatting visualization
- COLMAP-format camera pose exports for downstream 3D Gaussian Splatting training

The study sequences are two street-level drives through **Tokyo, Japan** (Sequences A and B), totaling ~1,740 raw Mapillary frames sampled at stride 5 down to ~174 inference frames each.

---

## Notebooks

| Notebook | Environment | Purpose |
|---|---|---|
| [`mapanything-pipeline.ipynb`](notebooks/mapanything-pipeline.ipynb) | Local / Jupyter | Base pipeline — Mapillary API fetch, haversine trajectory metrics, MapAnything inference, Plotly 3D visualization |
| [`mapanything-pipeline-v2.ipynb`](notebooks/mapanything-pipeline-v2.ipynb) | Local / Jupyter | Extends base with Section 9: colored point cloud builder, Plotly 3D scatter (3σ clipped), COLMAP export |
| [`mapanything_pipeline-splatv1.ipynb`](notebooks/mapanything_pipeline-splatv1.ipynb) | Google Colab A100 | First splat attempt — naive `.splat` export (fixed scale=0.05, opacity=200), stride=2 inference |
| [`mapanything_pipeline-splatv2.ipynb`](notebooks/mapanything_pipeline-splatv2.ipynb) | Google Colab A100 | Capstone splat notebook — all refinements baked in (see below) |

### What changed from splatv1 → splatv2

| Change | Detail |
|---|---|
| **Stride 1 inference** | All ~174 downloaded frames used (vs. stride=2 = ~87 frames) |
| **3-split inference** | 174 frames split into 3 overlapping ~60-frame chunks with `memory_efficient_inference=False` to avoid 16+ GB OOM on A100 |
| **VRAM management** | Dedicated mem-clear cell; `del views, preds` after each split; explicit `gc.collect()` + `cuda.empty_cache()` |
| **Confidence extraction** | `conf` now extracted per-frame, `ndarray.ptp()` replaced with `.max()-.min()` (NumPy 2.0 compat), normalized to [0,1] |
| **kNN-adaptive scale** | `scipy.spatial.cKDTree` mean distance to 4 nearest neighbors per point → scale replaces fixed 0.05 |
| **Confidence-driven opacity** | Per-Gaussian opacity mapped `conf → [120, 255]` instead of fixed 200 |
| **Consistent 4M subsample** | `pts`, `cols`, `confs` indexed with the same `idx` before export — prevents shape mismatch crashes |
| **cloudflared viewer** | CORS-aware HTTP server + trycloudflare.com tunnel → direct `antimatter15.com/splat` link, no ngrok interstitial |
| **COLMAP export** | `cameras.txt` / `images.txt` / `points3D.txt` using model intrinsics; ready for `gaussian-splatting` trainer |

---

## Architecture

```
Mapillary API
    │  sequence IDs, geometry, computed_geometry (GPS poses)
    │  stride=5 download → ~174 JPEGs cached to Drive
    ▼
MapAnything (facebook/map-anything-apache)
    │  feed-forward transformer, no per-scene optimization
    │  memory_efficient_inference=False (full attention per split)
    │  outputs per frame:
    │    pts3d        (H×W×3)  — 3D points in world coords
    │    img_no_norm  (H×W×3)  — preprocessed image, pixel-aligned with pts3d
    │    conf         (H×W)    — per-pixel confidence
    │    mask         (H×W×1)  — validity mask
    │    camera_poses (4×4)    — cam2world
    │    intrinsics   (3×3)    — recovered pinhole K
    ▼
extract_reconstruction()
    │  confidence-masked points + colors + normalized conf
    │  5-value return: (pts, cols, confs, cam_poses, cam_K)
    ▼
    ├─── Plotly 3D scatter (30k subsample, 3σ clip, dark theme)
    ├─── Colored PLY  →  Open3D visualization
    ├─── Improved .splat  →  antimatter15.com/splat (browser viewer)
    └─── COLMAP export  →  gaussian-splatting trainer
```

---

## Key Technical Notes

### Stride math
- Mapillary Sequence B: **870 raw frames** in the API
- Downloaded at `stride=5` → **174 JPEGs** cached to Drive
- Inference at `stride=1` → all 174 used
- The "stride" in `SEGMENT_IDS[::stride]` applies to already-downloaded IDs — it does not re-stride the raw API

### Memory budget (A100 80GB)
- MapAnything model: ~12 GB VRAM
- 174 frames at `memory_efficient_inference=False`: O(N²) attention → ~17 GB needed per split
- 3-split approach (~60 frames each, 5-frame overlap): fits comfortably
- Free VRAM before inference should read ~65+ GB

### .splat format
32 bytes per Gaussian: `xyz (12) | scale (12) | rgba (4) | rotation_quaternion (4)`

| Version | Scale | Opacity | File size | Gaussians |
|---|---|---|---|---|
| Naive (splatv1) | Fixed 0.05 | Fixed 200 | ~6 MB | ~200k |
| Improved (splatv2) | kNN adaptive | Conf-weighted | ~128 MB | 4M |

Browser sweet spot: **3–5M Gaussians** (~96–160 MB). The 32.5M unsubsampled export (992 MB) crashes `antimatter15.com/splat`.

### COLMAP output structure
```
seq_B_colmap/
├── images/               # JPEGs copied and renamed 00001.jpg … 00174.jpg
└── sparse/0/
    ├── cameras.txt       # 1 PINHOLE camera with model-recovered fx, fy, cx, cy
    ├── images.txt        # QW QX QY QZ TX TY TZ per frame (scipy Rotation)
    └── points3D.txt      # intentionally empty — 3DGS initializes from random Gaussians
```

Train with:
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting
pip install -r requirements.txt
python train.py -s /path/to/seq_B_colmap --iterations 7000
```

---

## Setup

### Local (pipeline + v2 notebooks)
```bash
pip install requests pandas numpy plotly open3d scipy geopy
# Mapillary API token required — set MLY_TOKEN in notebook config cell
```

### Google Colab A100 (splat notebooks)
```python
# Cell 1 — install MapAnything
pip install "mapanything[all] @ git+https://github.com/facebookresearch/map-anything.git@main"

# Cell 2 — mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

Sequence images are expected at:
```
/content/drive/MyDrive/Mapillary_HERE_Init_Sequences/
├── seq_A_jn2K5gYsXOk46FtZrWAwMH/   # ~174 JPEGs
└── seq_B_rO5jKtQfFyvpEqGHYPLAb2/   # ~174 JPEGs
```

---

## Results

### Trajectory Metrics (Sequence B)
Haversine-derived metrics comparing GPS ground truth (`geometry`) against MapAnything-recovered poses (`computed_geometry`).

| Metric | Value |
|---|---|
| ATE (mean) | computed from haversine over full sequence |
| RPE (mean) | computed from frame-to-frame delta |
| Total distance | ~X km across ~174 frames |

### Point Cloud
- **~32.5M raw Gaussians** from 174 frames at full resolution
- **4M exported** after confidence-weighted subsampling
- Scale range: `0.005 – 0.5` (kNN-adaptive)
- Opacity range: `[120, 255]` (confidence-weighted)

---

## Roadmap

- [ ] Replace percentile confidence threshold with `non_ambiguous_mask` from pred dict
- [ ] Apply `metric_scaling_factor` per split to remove inter-split seam offsets
- [ ] Feed COLMAP output into full 3DGS trainer (Kerbl et al.)
- [ ] Compare MapAnything poses vs VGGT poses on ATE/RPE
- [ ] MLflow experiment tracking for model comparison runs

---

## References

- [MapAnything (Meta)](https://github.com/facebookresearch/map-anything) — feed-forward metric 3D reconstruction
- [Mapillary API v4](https://www.mapillary.com/developer/api-documentation) — street-level imagery + GPS sequences
- [3D Gaussian Splatting (Kerbl et al.)](https://github.com/graphdeco-inria/gaussian-splatting) — real-time neural rendering
- [antimatter15 splat viewer](https://antimatter15.com/splat/) — browser-based `.splat` visualization
- [cloudflared](https://github.com/cloudflare/cloudflared) — free tunnel for serving files from Colab

---

## License

Research use. MapAnything model weights are released under the Apache 2.0 license by Meta. Mapillary imagery is subject to [Mapillary Terms of Service](https://www.mapillary.com/terms).
