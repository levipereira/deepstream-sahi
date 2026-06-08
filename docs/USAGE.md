# Usage Guide

This guide explains how to run the DeepStream SAHI test pipelines, compare results, and customize parameters.

> **Prerequisite:** Complete the [Installation Guide](INSTALL.md) first. All commands below assume you are inside the DeepStream container with the `pyds` virtualenv activated:
> ```bash
> source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate
> ```

## Download Test Videos

The test videos are hosted on Google Drive (not included in the repository). Download them and place them in the `python_test/videos/` directory:

**[Download test videos from Google Drive](https://drive.google.com/drive/folders/1CRLnuH9AtTwmxRz7z-Mtu6ErKx__VMK4)**

| File | Size | Scene |
|------|------|-------|
| `aerial_crowding_01.mp4` | 274.6 MB | Dense pedestrian crowd (2560x1440) |
| `aerial_crowding_02.mp4` | 20.1 MB | Very dense crowd with motorcycles (2560x1440) |
| `aerial_vehicles.mp4` | 8.5 MB | Dense vehicle traffic (2560x1440) |

```bash
# Place the downloaded .mp4 files in:
/apps/deepstream-sahi/python_test/videos/
```

> **Tip:** You can also use your own videos. Just pass the file path with `-i` / `--input` to the test scripts.

## Directory Layout

```
python_test/
├── common/                         # Shared GStreamer utilities
├── deepstream-test-sahi/
│   ├── config/
│   │   ├── pgie/                   # nvinfer configs (per model)
│   │   ├── preprocess/             # nvsahipreprocess configs (per resolution)
│   │   └── tracker/                # NvDCF tracker config
│   ├── models/                     # ONNX models, TensorRT engines, label files
│   ├── deepstream_test_sahi.py     # SAHI pipeline (sliced inference)
│   ├── deepstream_test_no_sahi.py  # Standard pipeline (full-frame baseline)
│   ├── pipeline_common.py          # Shared pipeline code and model registry
│   ├── compare_results.py          # CSV comparison and report generation
│   └── requirements_compare.txt    # Dependencies for compare_results.py
└── videos/                         # Input test videos (download separately)
```

## Available Models

| Model ID | Description | Input Size | Classes | Output / parser |
|----------|-------------|-----------|---------|-----------------|
| `visdrone-full-640` | VisDrone GELAN-C (full-frame training) | 640x640 | 11 | EfficientNMS → `NvDsInferYoloNMS` |
| `visdrone-sliced-448` | VisDrone GELAN-C (sliced training) | 448x448 | 11 | EfficientNMS → `NvDsInferYoloNMS` |
| `visdrone-yolo26s-sliced-448` | VisDrone YOLO26s (sliced training, end-to-end / NMS-free) | 448x448 | 11 | `[N,6]` → `NvDsInferYoloE2E` |
| `visdrone-yolo26n-sliced-416` | VisDrone YOLO26n (sliced training, end-to-end / NMS-free) | 416x416 | 11 | `[N,6]` → `NvDsInferYoloE2E` |

See **[SAHI Model Benchmark](SAHI_MODEL_BENCHMARK.md)** for a real in-pipeline FPS + detection comparison.

## How Model Configuration Differs: SAHI vs Standard

This is the most important architectural difference between the two pipelines. Understanding it is essential for adding new models.

### Standard Pipeline (No SAHI)

```
nvstreammux → nvinfer → nvdsosd → sink
```

`nvinfer` receives raw frames from `nvstreammux` and **handles everything internally**: scaling to network input size, color conversion, normalization, and tensor preparation. All model parameters live in a single config file (the **pgie config**):

```ini
# config/pgie/visdrone-full-640.txt — nvinfer does all preprocessing
[property]
net-scale-factor=0.0039215697906911373   # 1/255 normalization
infer-dims=3;640;640                     # network input C;H;W
batch-size=16
onnx-file=../../models/model.onnx
model-engine-file=../../models/model.engine
# ... detection params, custom parser, etc.
```

### SAHI Pipeline

```
nvstreammux → nvsahipreprocess → nvinfer (input-tensor-meta=1) → nvsahipostprocess → nvdsosd → sink
```

With SAHI, preprocessing is **split between two components**:

1. **`nvsahipreprocess`** takes over the role of scaling, cropping, and tensor preparation. It slices each frame into overlapping tiles, scales each tile to the network input size, and produces ready-to-infer GPU tensors. These parameters must be configured in the **preprocess config file**.

2. **`nvinfer`** receives pre-prepared tensors via `input-tensor-meta=1` and **skips its own preprocessing entirely**. It only runs the inference engine and output parsing.

This means the model's input dimensions, color format, normalization, and tensor layout must be specified **in the preprocess config**, not just in the pgie config:

```ini
# config/preprocess/preprocess_640.txt — nvsahipreprocess handles preprocessing
[property]
processing-width=640                         # must match network input
processing-height=640                        # must match network input
network-input-shape=16;3;640;640             # B;C;H;W — batch, channels, height, width
network-input-order=0                        # 0=NCHW
network-color-format=0                       # 0=RGB
tensor-data-type=0                           # 0=FP32
tensor-name=images                           # must match model's input layer name
maintain-aspect-ratio=1
symmetric-padding=1
custom-lib-path=/.../libcustom2d_preprocess.so
custom-tensor-preparation-function=CustomTensorPreparation

[user-configs]
pixel-normalization-factor=0.003921568       # 1/255 — same as net-scale-factor in pgie
```

### What Must Match Between the Two Configs

When adding a new model for SAHI, these parameters must be consistent across both config files:

| Parameter | pgie config (`nvinfer`) | preprocess config (`nvsahipreprocess`) |
|-----------|------------------------|---------------------------------------|
| Input dimensions | `infer-dims=3;640;640` | `processing-width=640`, `processing-height=640`, `network-input-shape=16;3;640;640` |
| Normalization | `net-scale-factor=0.003921568` | `pixel-normalization-factor=0.003921568` (in `[user-configs]`) |
| Color format | `model-color-format=0` (RGB) | `network-color-format=0` (RGB) |
| Batch size | `batch-size=16` | `network-input-shape=16;...` (first dimension) |
| Input tensor name | (implicit) | `tensor-name=images` |
| GIE unique ID | `gie-unique-id=1` | `target-unique-ids=1` |

> **Key point:** If you change the model's input size (e.g. from 640 to 448), you must update **both** the pgie config (`infer-dims`) **and** create/use the matching preprocess config (`processing-width`, `processing-height`, `network-input-shape`).

### Config Files in This Project

| Model | pgie config | preprocess config |
|-------|------------|-------------------|
| `visdrone-full-640` | `config/pgie/visdrone-full-640.txt` | `config/preprocess/preprocess_640.txt` |
| `visdrone-sliced-448` | `config/pgie/visdrone-sliced-448.txt` | `config/preprocess/preprocess_448.txt` |
| `visdrone-yolo26s-sliced-448` | `config/pgie/visdrone-yolo26s-sliced-448.txt` | `config/preprocess/preprocess_448.txt` (shared) |
| `visdrone-yolo26n-sliced-416` | `config/pgie/visdrone-yolo26n-sliced-416.txt` | `config/preprocess/preprocess_416.txt` |

## Adding a New Model (worked example: YOLO26)

The repo ships the **YOLO26s** detector (`visdrone-yolo26s-sliced-448`) as a second-family example.
Use these steps to add any new detector to the SAHI pipeline.

**1. Place the ONNX** under `models/` (Git LFS). YOLO26 ships in the model package; it is symlinked here:

```bash
# from python_test/deepstream-test-sahi/models/
ln -sf ../../../model_zoo/visdrone_yolo26/04_models/yolo26s/best.onnx \
       yolo26s-visdrone-sliced-448.onnx
# (or copy the .onnx if you prefer a standalone file)
```

**2. Pick the parser** based on the model's output tensor (all live in `libnvds_infer_yolo.so`):

| Model output | `parse-bbox-func-name` | Notes |
|--------------|------------------------|-------|
| EfficientNMS tensors (`num_dets`, `det_boxes`, ...) | `NvDsInferYoloNMS` | GELAN-C default |
| End-to-end / NMS-free single `[N,6]` (x1,y1,x2,y2,conf,class) | `NvDsInferYoloE2E` | YOLO26 / YOLOv10 / YOLO-NAS |
| Instance-seg | `NvDsInferYoloMask` | seg models |

**3. Write the pgie config** (`config/pgie/<model>.txt`). For YOLO26 see
`config/pgie/visdrone-yolo26s-sliced-448.txt`. Key fields vs the GELAN config:
`parse-bbox-func-name=NvDsInferYoloE2E`, `output-blob-names=output0`, `num-detected-classes=11`,
and **`cluster-mode=2`** (always — `cluster-mode=4` renders wrong boxes in the OSD here).

**4. Reuse or create the preprocess config.** YOLO26s is 448×448, so it **reuses**
`config/preprocess/preprocess_448.txt`. For a different input size, copy it and update
`processing-width/height` and `network-input-shape` (see the matching-parameters table above).
Set `tensor-name` to match the model's input layer name (YOLO models use `images`).

**5. Register the model** in `pipeline_common.py` → `MODELS` (id, `pgie_config`, `preprocess_config`,
`input_size`, `class_names`, `default_slice`). It then appears in `--model` automatically.

**6. Run it** (same CLI as any model):

```bash
python3 deepstream_test_sahi.py --model visdrone-yolo26s-sliced-448 --csv -i ../videos/aerial_vehicles.mp4
```

> The first run builds the TensorRT engine from the ONNX (cached as
> `models/<name>.engine`); subsequent runs reuse it.

### Non-obvious rules when adding models

- **`cluster-mode=2` always.** Even for NMS-free models, `cluster-mode=4` renders **wrong boxes in the OSD** (the NVIDIA clustering path normalizes coordinates the downstream stages rely on).
- **`batch-size` (pgie) = `network-input-shape[0]` (preprocess) = tiles/frame.** nvinfer runs `ceil(tiles/batch)` inference passes per frame; matching them → 1 frame = 1 inference (roughly 3× FPS, no accuracy change). Tiles/frame at 2560×1440: **41** @ slice 416, **29** @ 448, **16** @ 640. Adjust for your resolution/slice size.
- **Multi-camera:** batch = **`cameras × tiles/frame`** (the preprocess emits that many ROIs). For example, 2 cameras × 41 tiles = batch 82 (heavy); aim for a low tile count so `cameras × tiles` stays manageable.
- **Object overload:** above ~2000 objects/frame the OSD + GreedyNMM merge dominate and FPS collapses. The OSD probe emits a one-time WARN. Bound it with `pre-cluster-threshold`, fewer tiles, or `max-detections` on `nvsahipostprocess`.

## Running the SAHI Pipeline

The SAHI pipeline uses `nvsahipreprocess` to dynamically slice each frame, `nvinfer` for inference on each slice, and `nvsahipostprocess` (GreedyNMM) to merge duplicate detections.

```
uridecodebin → nvstreammux → nvsahipreprocess → nvinfer → nvsahipostprocess → nvdsosd → sink
```

### Basic Usage

```bash
cd /apps/deepstream-sahi/python_test/deepstream-test-sahi

python3 deepstream_test_sahi.py --model visdrone-full-640 -i ../videos/aerial_crowding_01.mp4
```

### With Display (requires X11 / WSLg)

```bash
python3 deepstream_test_sahi.py --model visdrone-full-640 --display -i ../videos/aerial_crowding_01.mp4
```

> By default, the pipeline uses `fakesink` (no display window). Use `--display` only when a display server is available. On WSL2 without X11 forwarding, `--display` will be ignored with a warning and the pipeline falls back to `fakesink` automatically.

### With Object Tracking

```bash
python3 deepstream_test_sahi.py --model visdrone-sliced-448 --tracker -i ../videos/aerial_vehicles.mp4
```

### Save Output as MP4

```bash
python3 deepstream_test_sahi.py --model visdrone-full-640 \
    --output-mp4 results/output_sahi.mp4 \
    -i ../videos/aerial_crowding_01.mp4
```

### Export Per-Frame Detections to CSV

```bash
# Auto-generated filename in results/
python3 deepstream_test_sahi.py --model visdrone-full-640 --csv -i ../videos/aerial_vehicles.mp4

# Custom path
python3 deepstream_test_sahi.py --model visdrone-full-640 --csv results/my_test.csv -i ../videos/aerial_vehicles.mp4
```

### Custom SAHI Slice Parameters

```bash
python3 deepstream_test_sahi.py --model visdrone-sliced-448 \
    --slice-width 448 --slice-height 448 \
    --overlap-w 0.25 --overlap-h 0.25 \
    --match-metric 1 --match-threshold 0.5 \
    -i ../videos/aerial_crowding_01.mp4
```

### SAHI-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--slice-width` | model's input size | Slice width in pixels |
| `--slice-height` | model's input size | Slice height in pixels |
| `--overlap-w` | `0.2` | Horizontal overlap ratio (0.0–0.99) |
| `--overlap-h` | `0.2` | Vertical overlap ratio (0.0–0.99) |
| `--no-full-frame` | disabled | Omit the full frame as an extra slice |
| `--match-metric` | `1` (IoS) | NMM metric: `0` = IoU, `1` = IoS |
| `--match-threshold` | `0.5` | NMM overlap threshold for merging |
| `--validate-slices` | disabled | Print per-frame slice count validation |

## Running the Standard Pipeline (No SAHI)

The baseline pipeline runs full-frame inference without slicing — useful for quantifying the improvement that SAHI provides.

```
uridecodebin → nvstreammux → nvinfer → nvdsosd → sink
```

```bash
python3 deepstream_test_no_sahi.py --model visdrone-full-640 -i ../videos/aerial_crowding_01.mp4
```

All common arguments (`-i`, `--tracker`, `--display`, `--output-mp4`, `--csv`, `--resolution`) work the same as with the SAHI pipeline.

## Common Arguments (Both Pipelines)

| Argument | Default | Description |
|----------|---------|-------------|
| `-i` / `--input` | **required** | Path to video file (MP4, MKV, AVI, H264, ...) |
| `--model` | **required** | Model ID (see table above) |
| `--resolution` | `1440p` | Muxer preset: `4k`, `1440p`, `1080p`, `720p` |
| `--muxer-width` / `--muxer-height` | — | Custom muxer resolution (alternative to `--resolution`) |
| `--tracker` | disabled | Enable NvDCF object tracking |
| `--display` | disabled | Enable display sink (EGLSink / nv3dsink) |
| `--output-mp4 PATH` | — | Save H264-encoded MP4 |
| `--bitrate` | `15000000` | H264 encoder bitrate in bps (15 Mbps, optimized for 2K) |
| `--csv [PATH]` | — | Export per-frame detections to CSV |

## Comparing Results

After running both pipelines with `--csv`, use `compare_results.py` to generate comparison reports with charts.

### Install Dependencies

```bash
pip install -r requirements_compare.txt
```

### Run Comparison

```bash
python3 compare_results.py \
    results/video_model_sahi_timestamp.csv \
    results/video_model_nosahi_timestamp.csv \
    -a "SAHI" -b "No SAHI" \
    -o results/comparison_sahi_vs_nosahi
```

This generates:
- `report.md` — Markdown summary with statistics
- `01_total_objects_over_frames.png` — Detection count per frame
- `02_class_comparison_bar.png` — Mean detections per class
- `03_total_objects_histogram.png` — Detection count distribution
- `04_difference_over_frames.png` — Per-frame difference
- `05_top_classes_over_time.png` — Top classes over time

## Example: Full Evaluation Workflow

```bash
cd /apps/deepstream-sahi/python_test/deepstream-test-sahi
VIDEO=../videos/aerial_crowding_01.mp4
MODEL=visdrone-full-640

# 1. Run SAHI pipeline (fakesink is the default — no --no-display needed)
python3 deepstream_test_sahi.py --model $MODEL --csv -i $VIDEO

# 2. Run standard pipeline
python3 deepstream_test_no_sahi.py --model $MODEL --csv -i $VIDEO

# 3. Compare results (use the CSV filenames generated above)
python3 compare_results.py \
    results/aerial_crowding_01_${MODEL}_sahi_*.csv \
    results/aerial_crowding_01_${MODEL}_nosahi_*.csv \
    -a "${MODEL}-sahi" -b "${MODEL}-no-sahi"
```
