# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**DeepStream-SAHI** — GStreamer plugins that bring [SAHI](https://github.com/obss/sahi) tiled
inference to NVIDIA DeepStream, keeping slicing, inference, and the cross-tile merge **inside** the
pipeline:

```
nvstreammux → nvsahipreprocess → nvinfer (input-tensor-meta) → nvsahipostprocess → [nvtracker] → nvdsosd → sink
```

- `nvsahipreprocess` — computes slices per frame, GPU-crops+rescales them into the network input, and
  feeds `nvinfer` as a batched tensor (`input-tensor-meta=1`).
- `nvsahipostprocess` — merges duplicate detections from overlapping tiles with a two-phase GreedyNMM
  (spatial-grid accelerated).
- `libnvds_infer_yolo.so` (`deepstream_source/libs/nvdsinfer_yolo/`) — custom nvinfer bbox parsers:
  `NvDsInferYoloNMS` (EfficientNMS tensors), `NvDsInferYoloE2E` (`[N,6]` end-to-end),
  `NvDsInferYoloMask` (seg).

The repo ships multiple detector families for VisDrone (11 classes) so a user can **pick a model and
run** — see "Models" below and `docs/USAGE.md`.

## Environment (always inside the DeepStream container)

Everything builds/runs inside the DeepStream container — there is no host build. CUDA, TensorRT,
GStreamer, `pyds` all live there.

```bash
docker run -it --gpus all --net=host -v `pwd`:/apps/deepstream-sahi -w /apps/deepstream-sahi \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch
# inside:
./install.sh                       # builds plugins + libnvds_infer_yolo.so + pyds (first time, slow)
./install.sh --plugins-only        # rebuild ONLY the SAHI plugins + parser (fast; after C++ edits)
source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate
cd python_test/deepstream-test-sahi
python3 deepstream_test_sahi.py --model <model> --no-display --csv -i ../videos/<video>.mp4
```

- DeepStream **9.0** (CUDA 13.1, TRT 10.14) and **8.0** are supported; `install.sh` detects the version.
- `*.onnx` are **Git LFS** (`git lfs install` before clone/pull). `*.engine` are git-ignored
  (built on first run, cached next to the ONNX).
- Test videos are not in the repo — download from the Google Drive link in `docs/USAGE.md` into
  `python_test/videos/`.

## Build / test loop for C++ plugin changes

1. Edit `deepstream_source/gst-plugins/gst-nvsahi{pre,post}process/` or `libs/nvdsinfer_yolo/`.
2. `./install.sh --plugins-only` (≈1–2 min; `-Wall -Werror -O2`).
3. Re-run the pipeline. The TensorRT engine is cached, so iteration is fast.
> There is **no host compiler path** for these — they need the DeepStream SDK headers in the container.

## Models (pick one with `--model`)

Registered in `python_test/deepstream-test-sahi/pipeline_common.py` (`MODELS` dict). Each maps to a
pgie config (`config/pgie/`) + a preprocess config (`config/preprocess/`).

| Model id | Family | Output → parser | Notes |
|----------|--------|-----------------|-------|
| `visdrone-full-640`, `visdrone-sliced-448` | YOLOv9-C / GELAN | EfficientNMS → `NvDsInferYoloNMS` | original samples |
| `visdrone-yolo26n-sliced-416` | YOLO26 (e2e, NMS-free) | `[N,6]` → `NvDsInferYoloE2E` | input layer `images`, n@416 |
| `visdrone-yolo26s-sliced-448` | YOLO26 (e2e, NMS-free) | `[N,6]` → `NvDsInferYoloE2E` | input layer `images`, s@448 |

Trained weights + provenance live in `model_zoo/visdrone_yolo26/` (Part 1: training +
benchmark). **The models here are pre-trained.** To *re-train*, use the **original upstream repos**
(see "Training" below) — this repo only deploys.

## Non-obvious rules (READ before editing configs)

- **`cluster-mode=2` always.** Even for NMS-free models, `cluster-mode=4` renders **wrong boxes in the
  OSD** (the NVIDIA clustering path normalizes coordinates the downstream stages rely on). Every pgie
  config uses `cluster-mode=2`.
- **`batch-size` (pgie) = `network-input-shape[0]` (preprocess) = tiles/frame.** nvinfer runs
  `ceil(tiles/batch)` inference passes per frame; matching them → 1 frame = 1 inference (≈3× FPS, no
  accuracy change). Tiles/frame @ 2560×1440: **41** @ slice 416, **29** @ 448, **16** @ 640. Adjust
  for your resolution/slice.
- **Multi-camera:** batch = **`cameras × tiles/frame`** (the preprocess emits that many ROIs). 2 cams
  × 41 tiles = batch 82 (heavy); aim for a **low tile count** so `cameras × tiles` stays viable
  (e.g. 2 × 12 = 24). Balance the model's **training resolution** vs slice size — a model trained at a
  higher input lets you use larger 1:1 slices → fewer tiles with **no recall loss**. See
  `docs/SAHI_MODEL_BENCHMARK.md` → "Deployment planning".
- **Tiles may be larger than the network input** — the preprocess resizes each crop. Fewer/larger
  tiles = faster but lower small-object recall; **slice = network input (416) = max recall**.
- **Per-class output names** must match the parser: `NvDsInferYoloE2E` → `output-blob-names=output0`;
  `NvDsInferYoloNMS` → `num_dets;det_boxes;det_scores;det_classes`. YOLO/GELAN use
  `tensor-name=images`, letterbox (`maintain-aspect-ratio=1`).
- **Object overload:** above ~2000 objects/frame the OSD + GreedyNMM merge dominate and FPS collapses.
  The OSD probe emits a one-time WARN. Bound it with `pre-cluster-threshold`, fewer tiles, or
  `max-detections` on `nvsahipostprocess`.

## Performance summary (what's the bottleneck)

Measured on RTX 4090, FP16, `fakesink sync=false` — full data in `docs/SAHI_MODEL_BENCHMARK.md`.
- GPU inference is **not** the pipeline limiter for these light models (no-SAHI caps ~615 FPS = fixed
  per-frame overhead). **Tiling** (tiles/frame) and **detection volume** dominate.
- The `nvsahipostprocess` merge is ~0.18 ms/frame at the operating threshold — **not** the bottleneck.
  The grid `query()` was optimized to O(k) (visited-stamp; ~23% faster postprocess at extreme density,
  results identical).
- Real FPS (RTX 4090, FP16, `fakesink sync=false`): YOLO26n 135 → 389 FPS (after the batch fix),
  GELAN ≈76 FPS (full-frame).

## Documentation index

| Doc | What |
|-----|------|
| `docs/INSTALL.md` | container setup, plugin build |
| `docs/USAGE.md` | run the pipeline, CLI args, **add a new model** |
| `docs/SAHI_MODEL_BENCHMARK.md` | FPS/detection benchmark, bottleneck analysis, batch/tile tuning |
| `docs/PLUGIN_REVIEW.md` | nvsahipre/postprocess code review + optimizations |
| `docs/PLUGINS.md` | plugin property reference |
| `model_zoo/visdrone_yolo26/README.md` | Part 1: training + accuracy + TRT benchmark |
| `.claude/skills/` | repeatable workflows (export, build, benchmark) |

## Training (NOT in this repo — use upstream)

This repo **deploys** pre-trained models; it does not train. To reproduce/retrain:
- **YOLO26:** the official Ultralytics repo (`yolo detect train ...`), then `yolo export format=onnx
  dynamic=True simplify=True`.
- **YOLOv9-C/GELAN:** the upstream YOLOv9 repo (EfficientNMS export).
- Dataset (VisDrone sliced 416, 11 classes) + the exact training commands/hyperparameters are in
  `model_zoo/visdrone_yolo26/{01_dataset,02_training}/`.

## Conventions
- Multi-GPU runs: pin with `CUDA_VISIBLE_DEVICES=<n>` per `docker exec` (or per container) — the
  pgie config `gpu-id=0` refers to the *visible* device.
- Keep ONNX as **git symlinks** under `python_test/deepstream-test-sahi/models/` pointing at the
  `model_zoo/` LFS copies (no duplication; symlinks are stored as symlinks, not LFS blobs).
- Generated `results/` (CSV/MP4) and `*.engine` are git-ignored.
- Don't commit to `master`. Work on a feature branch; the maintainer squashes after testing.
