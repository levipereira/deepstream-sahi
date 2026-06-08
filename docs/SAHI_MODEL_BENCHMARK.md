# SAHI Pipeline Benchmark — YOLO26 vs YOLOv9-C (GELAN)

Real, in-pipeline FPS and detection comparison of two detector families running inside the DeepStream-SAHI
pipeline (slicing + inference + GreedyNMM merge), measured with a **non-throttled `fakesink`
(`sync=false`)** so the numbers reflect the pipeline's true maximum throughput, not the video frame
rate.

## Setup

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 4090 |
| Container | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` (TensorRT 10.14, CUDA 13.1) |
| Precision | FP16 (`network-mode=2`) |
| Sink | `fakesink sync=false async=false qos=false` (max-throughput, no frame drop) |
| Video | `aerial_crowding_01.mp4` — **15,794 frames, 2560×1440**, dense pedestrian crowd |
| SAHI | overlap 0.2/0.2, full-frame slice on; merge = GreedyNMM, metric IoS, threshold 0.5 |
| Clustering | `cluster-mode=2` (all models — required for correct OSD boxes) |
| FPS metric | median of the per-~5s PERF windows (engine-build/warm-up excluded) |

| Model | Params | Slice/input | Output → parser | pre-cluster-threshold |
|-------|-------:|:-----------:|-----------------|:---------------------:|
| YOLO26n | ≈2.5 M | 416 | `[N,6]` → `NvDsInferYoloE2E` (NMS-free) | 0.25 |
| YOLO26s | ≈9 M | 448 | `[N,6]` → `NvDsInferYoloE2E` (NMS-free) | 0.25 |
| YOLOv9-C (GELAN) | ≈25 M | 448 | EfficientNMS → `NvDsInferYoloNMS` | 0.25 |

## FPS (real, sync=false)

| Model | Median FPS | Mean | Min–Max | Relative |
|-------|-----------:|-----:|:-------:|:--------:|
| **YOLO26n** (416) | **135.6** | 135.9 | 77–201 | 1.00× (fastest) |
| **YOLOv9-C / GELAN** (448) | **76.5** | 75.8 | 73–77 | 0.56× |

## Detections (whole video, 15,794 frames)

| Model | Total objects | Mean / frame | Max / frame | Top classes (mean/frame) |
|-------|--------------:|-------------:|------------:|--------------------------|
| YOLOv9-C / GELAN (thr 0.25) | 1,412,676 | 89.4 | 335 | pedestrian 74, people 7, motor 4, car 2 |
| YOLO26n (thr 0.25) | 922,136 | 58.4 | 265 | pedestrian 51, car 3, people 2, motor 1 |

**Total-objects difference:** GELAN finds **+53 %** more objects than YOLO26n
(1.41 M vs 0.92 M) over the clip, largely due to its larger capacity (25 M vs 2.5 M params).

## Reading the results

- **Speed:** YOLO26n is the fastest by a wide margin (135 FPS) — tiny model, NMS-free head, light
  postprocess. GELAN is mid (76 FPS) — its 25 M parameters make it model-bound even at 1 inference/frame.
- **Detections:** GELAN recovers more objects than YOLO26n on this dense aerial scene (~1.5× at the
  same threshold), with a stronger bias toward non-pedestrian classes (motor, car). YOLO26n detections
  are pedestrian-dominated but at a significantly lower compute cost.
- **Operating-point summary:** YOLO26n for maximum FPS with a lean model; GELAN when detection
  completeness on a wider set of classes matters and ~76 FPS is sufficient; GELAN is ~10× the params
  of YOLO26n.

## Reproduce

```bash
# inside the DeepStream container, pyds venv active
cd python_test/deepstream-test-sahi
for M in visdrone-yolo26n-sliced-416 visdrone-sliced-448; do
  python3 deepstream_test_sahi.py --model "$M" --no-display --csv -i ../videos/aerial_crowding_01.mp4
done
# FPS: median of the PERF lines; detections: results/*.csv (total_objects column)
```

> Numbers depend on threshold, slice size/overlap, and GPU. They were captured on a single RTX 4090
> with the shipped configs; tune `pre-cluster-threshold` and the SAHI `--match-threshold` for your
> scene before drawing final conclusions.

---

# Where the bottleneck is (empirical isolation) and how to optimize

Controlled experiments on the same RTX 4090, 2560×1440, FP16, `fakesink sync=false`, to separate
**model** cost from **tiling** cost from **postprocess** cost.

## Isolation experiments

| Scenario | YOLO26n (416) | YOLOv9-C/GELAN (448) |
|----------|--------------:|---------------------:|
| **No-SAHI** (1 inference/frame, 0 det) | 617 fps | **313 fps** |
| **SAHI on black** (all tiles, 0 det) | 207 (41t) | 71.9 (29t) |
| **SAHI real crowd** (with detections) | 135.6 (58 obj) | 76.5 (89 obj) |

Findings:
- **No-SAHI:** YOLO26n is capped at ~615 fps → at 1 inference/frame it is bound by the **fixed
  per-frame pipeline overhead** (decode + mux + convert + OSD + sink), *not* the model. GELAN at
  313 fps is **model-bound** (its 25 M weights show: ~2× slower per inference).
- **SAHI black vs no-SAHI:** tiling is the #1 throughput killer (1→41 tiles: 617→207 for YOLO26n).
  It multiplies inference + the per-tile pipeline work.
- **SAHI real vs black:** GELAN 72→76 (no change → EfficientNMS postprocess ≈ free). The postprocess
  cost for YOLO-E2E `[N,6]` models scales with detection count; raising `pre-cluster-threshold`
  reduces objects/frame and recovers FPS.

## 1. Preprocess (tiling)
`nvsahipreprocess` crops+rescales all tiles in **one batched `NvBufSurfTransformAsync`** (GPU, async),
so it is not the dominant cost — the cost is the *number of tiles* (it drives how much inference and
per-tile work follows). The plugin has no built-in timing counter (unlike `nvsahipostprocess`); the
GST latency tracer does not emit per-element times for the DeepStream elements. **Lever: fewer tiles.**

## 2. Batch size = tiles/frame (biggest free win)
nvinfer must run `ceil(tiles / batch-size)` inference passes per frame; each pass has fixed
enqueue/sync overhead. With 41 tiles and `batch-size=16` that is **3 passes/frame**. Setting
**`batch-size = tiles/frame`** makes it **1 frame = 1 inference**:

| YOLO26n config (real crowd) | passes/frame | FPS |
|-----------------------------|:------------:|----:|
| 41 tiles @416, batch 16 | 3 | 135 |
| 41 tiles @416, **batch 41** | 1 | **389** |
| 16 tiles @640→416, batch 16 | 1 | **725** |

- **batch = tiles → ~3× FPS with ZERO accuracy change** (same tiles, same detections). Free win.
- Tiles may be **larger than the network input** — `nvsahipreprocess` resizes each crop to
  `processing-width/height`. So you can use **fewer, larger tiles** (e.g. 640→416, 16 tiles) for
  another ~2× FPS, at the cost of recall on small objects.
- **Rule:** set `batch-size` (pgie) and `network-input-shape[0]` (preprocess) **equal to the number
  of tiles per frame** for your resolution/slice. Tiles/frame here (2560×1440): **41** @ slice 416,
  **16** @ slice 640, **29** @ slice 448. The shipped 416 configs use `batch-size=41`.

## 3. Postprocess
- **GELAN / EfficientNMS models:** postprocess is essentially free (NMS runs in the engine on GPU,
  tiny fixed output). Nothing to do.
- **YOLO-E2E `[N,6]` models (YOLO26):** cost scales with the number of detections (CPU parse of the
  surviving rows + per-object meta + OSD + GreedyNMM). Levers, in order of impact:
  1. **Raise `pre-cluster-threshold`** to reduce detections/frame and keep the merge fast.
  2. **`class-agnostic=true`** on `nvsahipostprocess` (more suppression, less work).
  3. **Cap `max-detections`** / lower `topk` to bound worst-case frames.
  4. **Less overlap** / fewer tiles → fewer duplicate boxes to merge.

**Measured follow-up — the postprocess is NOT the bottleneck at normal thresholds.** Profiling
`nvsahipostprocess` at the operating point shows only **~0.18 ms/frame** (≈3.5% of the budget at
~196 FPS). The dominant costs are **tiling** (tiles/frame) and **detection volume** at low thresholds,
not the cross-tile merge itself. The `nvsahipostprocess` grid `query()` was optimized to O(k)
(visited-stamp; ~23% faster at extreme detection density, results identical) — this only pays off
at very low thresholds. The real levers are **tile count (slice size)** and **threshold**.

## Recommended starting point
1. `batch-size = tiles/frame` (free ~3×).
2. Tune `pre-cluster-threshold` to your scene density (0.25 is the YOLO26/GELAN default).
3. Pick slice size by the accuracy/speed trade-off (smaller = better small-object recall, slower).

---

# Tile-count sweep — how many tiles before detection drops

Slice size sets the tile count (fewer/larger tiles = faster). The crop is resized to the network
input, so **slice = input means 1:1, no downscale**; slice > input downscales each tile, shrinking
small objects. Sweep on YOLO26n (FP16, 2560×1440, batch = tiles), measuring mean detections/frame.

### Scene A — `aerial_crowding_01` (small, distant objects — the hard case)

| slice | tiles | obj/frame | vs slice 416 | FPS |
|------:|------:|----------:|:------------:|----:|
| **416** | **41** | **209.8** | **0%** | 196 |
| 512 | 25 | 152.8 | −27% | 289 |
| 576 | 19 | 112.5 | −46% | 377 |
| 640 | 16 | 100.1 | −52% | 424 |
| 768 | 13 | 77.0 | −63% | 484 |
| 896 | 9 | 52.9 | −75% | 580 |

→ **No plateau.** On small objects, recall drops immediately below 41 tiles (−27% already at 25).

### Scene B — `aerial_crowding_02` (denser, but larger/closer objects)

| slice | tiles | obj/frame | vs slice 416 |
|------:|------:|----------:|:------------:|
| 416 | 41 | 709 | 0% |
| 512 | 25 | 719 | ~0% |
| 576 | 19 | 697 | −2% |
| 640 | 16 | 654 | −8% |
| 768 | 13 | 546 | −23% |
| 896 | 9 | 389 | −45% |

→ **Flat to ~16 tiles**, knee at 13, collapse at 9.

### Conclusion
- **Maximum recall = slice at the model's input resolution (1:1, no downscale) = 41 tiles @ 416.**
  That is the max-recall point; it is the shipped default.
- The "best tile count" is **scene-dependent**: small/distant objects → keep 41 tiles (any reduction
  costs recall); larger/closer objects → you can drop to ~16 tiles for ~2× FPS nearly free.
- Reducing tiles is a **speed↔recall knob**, not a free lunch on hard scenes. Use the curve above to
  pick a point for your scene and FPS target.

---

# Deployment planning — slices × batch × cameras (the real trade-off)

Putting the pieces together: **more slices → more detections → more boxes → more OSD/merge work and a
bigger inference batch**. Fewer/larger slices are faster and cheaper but lose small-object recall. The
goal is the **fewest tiles that still meet your recall target**, then size the batch to them.

## Balance the model's *training resolution* against the slice size
The crop is resized to the network input, so a tile **at (or near) the model's training resolution is
1:1 — no downscale — and keeps full recall**. This means the lever is not just "smaller slices":

- A model trained at **416** wants 416 tiles for max recall → many tiles (41 @ 2560×1440).
- A model trained at a **higher input (e.g. 640)** lets you use **640 tiles at 1:1** → **far fewer
  tiles (16)** with *no* recall loss, because the larger crop isn't downscaled.

So **choosing/training a model at a larger input resolution can be a net win**: a slightly larger
slice drops you into the "fewer slices" regime without the recall penalty you'd pay by simply
upscaling 416 tiles. Match the model input to the slice you want to run.

## Batch must hold *cameras × tiles/frame*
DeepStream batches sources at `nvstreammux`, and `nvsahipreprocess` emits **`cameras × tiles_per_frame`**
ROIs into one inference tensor. For "1 inference per batched frame-set":

```
nvinfer batch-size = network-input-shape[0] = cameras × tiles_per_frame
```

This is the multi-camera constraint, and it is why tile count matters so much at scale:

| Cameras | Tiles/frame | Required batch | Viable? |
|:-------:|:-----------:|:--------------:|:-------:|
| 1 | 41 (slice 416) | 41 | yes |
| **2** | **41** | **82** | ✗ heavy (engine size/memory; may exceed a practical max batch) |
| 2 | **16** (slice 640) | **32** | ✓ |
| 2 | **12** (slice ~768) | **24** | ✓ ideal |
| 4 | 12 | 48 | ✓ |

**Recommendation:** for multi-camera, pick a model resolution + slice size that yields a **small
tile count per frame** (e.g. ~12–16), so `cameras × tiles` stays within a practical batch — then all
cameras' tiles infer in a single pass. A model trained at a higher resolution makes those larger
slices 1:1 (no recall loss), which is the ideal way to hit a low tile count.

> Single-source pipelines are validated here; multi-source uses the same ROI batching but has not been
> benchmarked end-to-end (see README → Limitations). The batch arithmetic above is the design rule.

---

# Roadmap / future improvements

### 1. Postprocess grid optimization — IMPLEMENTED (~23% faster at extreme density)
The `nvsahipostprocess` GreedyNMM spatial-grid `query()` was optimized to O(k) with a visited-stamp
(avoids re-visiting cells at extreme detection counts). Results are identical; the gain is only
visible at very low thresholds / high detection volumes. At normal operating thresholds the merge is
~0.18 ms/frame and not the bottleneck.

### 2. Other levers (available now)
- `batch-size = tiles/frame` — already applied (free ~3×).
- `pre-cluster-threshold` tuning — already applied per model.
- `class-agnostic=true` on `nvsahipostprocess`, `max-detections` cap, less overlap — for dense scenes.
- Slice size as the speed↔recall knob (see tile-count sweep above).
- Multi-stream (the GPU is underutilised at 1 stream) for aggregate throughput.

### 3. Nice-to-have
- Add a PERF timing counter to `nvsahipreprocess` (the postprocess already has one) to report tiling
  ms/frame directly.
- Per-resolution helper to auto-set `batch-size`/`network-input-shape` = computed tiles/frame.
- Tune slice/tile configuration per scene class distribution (small vs large objects).
