# Mapillary × MapAnything × VGGT-Long × Pi3 — Street-Level 3D Reconstruction Pipeline

A research pipeline for metric 3D reconstruction, trajectory evaluation, and Gaussian Splatting from Mapillary street-level imagery. Compares feed-forward transformer backbones on sequential drive datasets — no per-scene SfM required. All runs tracked via MLflow on DagsHub.

> Study area: street-level driving sequences through **Tokyo, Japan** — ~870 raw Mapillary frames per sequence, sampled at stride 5 down to ~174 inference frames, or stride 1 down to 100-frame dense subsets.

---

## Pipeline Overview

```
Mapillary API  ──►  Backbone Inference  ──►  COLMAP Export  ──►  Nerfstudio Splatfacto
  (GPS poses)       (MapAnything /             (cameras.txt        (trained 3DGS,
                     VGGT-Long / Pi3)           images.txt          30k steps,
                     metric 3D, no SfM)         points3D.txt)       ~123 MB PLY)
                          │
                          └──►  Trajectory Metrics (ATE / RPE / DTW)
                                MLflow → DagsHub
```

---

## Models Compared

| Backbone | Inference style | Chunk strategy | Notes |
|---|---|---|---|
| **MapAnything** | Feed-forward transformer | 3 splits, 5-frame overlap | Meta Apache 2.0. Strong baseline. |
| **VGGT-Long** | Feed-forward, chunked SIM3 alignment | Configurable chunk_size + overlap | Loop closure via SALAD. RPE_mean=2.46m on seq_B. |
| **Pi3** | Feed-forward | Single chunk preferred | HuggingFace `yyfz233/Pi3`. CONF_THRESHOLD=0.05 (conf maxes ~0.5). |
| **Pi-Long** | TBD | TBD | Planned. |

---

## Experiment Tracking

All runs logged to **DagsHub MLflow**:

Tracked params: `backbone`, `chunk_size`, `CONF_THRESHOLD`, `SAMPLE_RATIO`, `sequence_type`, `frame_count`, `stride`
Tracked metrics: `RPE_mean_m`, `scale_GPS_vision`, `DTW_normalised`, `pts_per_frame`, `ATE`

---

## Notebooks

| Notebook | Backbone | Purpose |
|---|---|---|
| `mapanything-pipeline` | MapAnything | Base pipeline — API fetch, haversine metrics, inference, Plotly 3D |
| `mapanything-pipeline-v2` | MapAnything | Adds PLY builder, COLMAP export |
| `mapanything_pipeline-splatv1` | MapAnything | First splat run (naive export, stride=2) |
| `mapanything_pipeline-splatv2` | MapAnything | Improved splat — kNN scale, conf-weighted opacity, COLMAP |
| `vggt_long_pipeline-2` | VGGT-Long | Full 174-frame sequence, 3 chunks, loop closure |
| `vggt_long_dense_subset` | VGGT-Long / Pi3 | Dense subset [0:100:1], checkpoint save/load, MLflow, COLMAP → splatfacto |

> Notebooks maintained separately. See [Notebooks](#notebooks) for access.

---

## Outputs

### Sequence Image Preview
*~174 street-level frames from a Tokyo driving sequence, fetched from the Mapillary API at stride 5.*

![Sequence preview](assets/sequence_preview.png)

---

### Mapillary GPS Trajectory
*Raw GPS track from `computed_geometry` overlaid on the route. Used as ground truth for ATE/RPE.*

![GPS trajectory](assets/gps_trajectory.png)

---

### Plotly 3D Point Cloud
*Dense colored point cloud reconstructed by MapAnything. 30k-point subsample, 3σ outlier clip, colored by `img_no_norm` pixel values.*

![Plotly 3D scatter](assets/plotly_pointcloud.png)

---

### Gaussian Splat — MapAnything Improved (splatv2)
*kNN-adaptive scale, confidence-weighted opacity [120–255], 4M Gaussians subsampled consistently. ~128 MB.*

![Improved splat](assets/splat_improved.png)

---

### Gaussian Splat — Nerfstudio Splatfacto (VGGT-Long / Pi3)
*Trained 3DGS via nerfstudio splatfacto (30k steps) initialized from VGGT-Long/Pi3 COLMAP export. ~123 MB. Forward-facing sequences produce spike artifacts — loop sequences planned.*

---

## Architecture

### MapAnything Branch
```
Mapillary API → MapAnything (3-split, ~60 frames/split, 5-frame overlap)
    │  pts3d (H×W×3), img_no_norm (H×W×3), conf (H×W), camera_poses (4×4), intrinsics (3×3)
    ▼
extract_reconstruction()
    ├── Plotly 3D scatter      30k subsample, 3σ clip
    ├── Colored PLY            Open3D
    ├── Improved .splat        4M Gaussians → antimatter15.com/splat
    └── COLMAP export          cameras.txt / images.txt / points3D.txt
```

### VGGT-Long / Pi3 Branch
```
Mapillary API → Subset [0:100:1] → VGGTLongTracked (chunk_records saved to .npy)
    │  world_points (T,H,W,3), world_points_conf (T,H,W), extrinsic (T,4,4), intrinsic (T,3,3)
    ▼
aggregate_world_points()  →  Plotly 3D / PLY
    ▼
COLMAP export (Cell A)    →  cameras.txt / images.txt / points3D.txt
    ▼
Nerfstudio splatfacto     →  30k steps, trained .ply → Drive
    ▼
MLflow logging            →  DagsHub
```

---

## Trajectory Metrics

| Metric | Description |
|---|---|
| **ATE** (Absolute Trajectory Error) | Mean positional deviation over the full sequence |
| **RPE** (Relative Pose Error) | Mean frame-to-frame delta error |
| **scale_GPS_vision** | Ratio of GPS-derived scale to model-derived scale |
| **DTW_normalised** | Dynamic Time Warping distance between predicted and GPS trajectory |

---

## Splat Comparison

| | MapAnything splatv1 | MapAnything splatv2 | VGGT-Long / Pi3 splatfacto |
|---|---|---|---|
| **Stride** | 2 | 1 (~174 frames) | 1 (100-frame subset) |
| **Scale** | Fixed 0.05 | kNN mean dist (4 neighbors) | Trained (nerfstudio) |
| **Opacity** | Fixed 200 | conf → [120, 255] | Trained |
| **Gaussians** | ~200k | 4M | Densified from seed pts |
| **File size** | ~6 MB | ~128 MB | ~123 MB |
| **Method** | Direct export | Direct export | Trained 3DGS (30k steps) |

---

## COLMAP Export

Standard output structure for downstream splatfacto / 3DGS training:

```
gsplat/
├── colmap_sparse/0/
│   ├── cameras.txt      # PINHOLE — intrinsics scaled to actual image res
│   ├── images.txt       # QW QX QY QZ TX TY TZ per frame
│   └── points3D.txt     # seed points from world_points (conf-filtered)
└── splatfacto.ply       # trained output
```

Train with nerfstudio:
```bash
ns-train splatfacto \
  --data /path/to/gsplat_scene \
  --output-dir /content/ns_output \
  --max-num-iterations 30000 \
  --pipeline.model.sh-degree 0 \
  colmap --colmap-path sparse/0 --images-path images
```

---

## Key Implementation Notes

**Run order (fresh Colab session)**
`§3 TF CPU-only → §1 Install → §2 Clone+Weights → §4 Drive mount → rest`
TF must be CPU-only before any import happens.

**Pi3 confidence scale**
Pi3 conf maxes at ~0.5, not ~5 like VGGT-Long. Set `CONF_THRESHOLD=0.05` for Pi3 runs.

**Pi3 chunk size**
Must override via `cfg.Data.chunk_size` immediately before `pipeline.run()` — Pi3 defaults override notebook variables.

**COLMAP camera resolution**
`cameras.txt` must use actual image dimensions (e.g. 518×392), not the model's internal output resolution (e.g. 574×434 for Pi3). Scale intrinsics: `fx_scaled = fx_model * (W_img / W_model)`.

**PyTorch 2.6 + nerfstudio**
Nerfstudio checkpoint loading breaks with PyTorch 2.6. Patch `eval_utils.py`:
```python
torch.load(load_path, map_location="cpu", weights_only=False)
```

**Drive FUSE — no symlinks**
Use `shutil.copy2` fallback. Subset images must be in `/content/` local storage, not Drive.

**OOM on A100 (MapAnything)**
174 frames at once exceeds VRAM with `memory_efficient_inference=False`. Use 3-split approach (~60 frames/split, 5-frame overlap).

**Forward-facing sequence limitation**
All sequences currently are forward-facing drives. Splatfacto produces spike artifacts due to poor lateral baseline and no loop closure. Next: Mapillary API search for loop sequences (start/end within ~50m).

**eval() string bug (VGGT-Long)**
OmegaConf parses float config values; VGGT-Long calls `eval()` on them. Fix:
```python
cfg.Model.IRLS.tol = str(IRLS_TOL)
cfg.Loop.SIM3_Optimizer.lambda_init = str(1e-6)
```

**NumPy 2.0 compatibility**
`ndarray.ptp()` removed. Replace with `.max() - .min() + 1e-8`.

---

## Roadmap

- [ ] Find Mapillary loop sequence (start/end within ~50m, Tokyo)
- [ ] Run splatfacto on VGGT-Long checkpoint (convert camera_poses.txt → COLMAP)
- [ ] Run MapAnything dense subset (stride=1, 100 frames) for fair three-way comparison
- [ ] MLflow sweep: CONF_THRESHOLD, chunk_size, backbone across same sequence
- [ ] Pi-Long backbone integration
- [ ] Investigate cold-start drift (first chunk, no prior context)
- [ ] YOLO-seg semantic filtering at loop closure / cold-start frames

---

## Setup

### Local (trajectory evaluation notebooks)
```bash
pip install requests pandas numpy plotly open3d scipy geopy
```

### Google Colab (reconstruction + splat notebooks)
```python
pip install nerfstudio plyfile ultralytics mlflow dagshub piexif tyro
```

Sequence images expected at:
```
MyDrive/Mapillary_HERE_Init_Sequences/
├── seq_A_<id>/images/    # ~174 JPEGs
└── seq_B_<id>/images/    # ~174 JPEGs
```

A Mapillary API token is required — set `MLY_TOKEN`. DagsHub token — set `DAGSHUB_USER_TOKEN`.

---

## References

- [MapAnything — Meta Research](https://github.com/facebookresearch/map-anything)
- [VGGT-Long](https://github.com/HengyiWang/VGGT-Long)
- [Pi3 — yyfz233/Pi3](https://huggingface.co/yyfz233/Pi3)
- [VGGT-Long-Gsplat](https://github.com/msilaev/VGGT-Long-Gsplat)
- [Nerfstudio splatfacto](https://docs.nerf.studio)
- [Mapillary API v4](https://www.mapillary.com/developer/api-documentation)
- [3D Gaussian Splatting (Kerbl et al.)](https://github.com/graphdeco-inria/gaussian-splatting)
- [antimatter15 splat viewer](https://antimatter15.com/splat/)

---

## License

Research use. MapAnything model weights are released under the Apache 2.0 license by Meta. Mapillary imagery is subject to [Mapillary Terms of Service](https://www.mapillary.com/terms).
