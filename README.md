# DeepStream SAHI

[![License](https://img.shields.io/badge/License-NVIDIA%20DeepStream%20EULA-76B900.svg)](https://developer.nvidia.com/deepstream-eula)
[![DeepStream](https://img.shields.io/badge/NVIDIA-DeepStream%208.0%20|%209.0-76B900?logo=nvidia)](https://developer.nvidia.com/deepstream-sdk)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.x-orange)](https://developer.nvidia.com/tensorrt)

GStreamer plugins that bring [SAHI](https://github.com/obss/sahi) slicing to NVIDIA DeepStream,
keeping slicing, inference, and the cross-tile merge **inside** the pipeline so it composes with
standard DeepStream components (tracking, analytics, brokers, display):

```text
nvstreammux -> nvsahipreprocess -> nvinfer -> nvsahipostprocess -> nvtracker -> nvdsosd
```

- `nvsahipreprocess` — computes per-frame slices, GPU-crops/rescales them, and feeds `nvinfer`.
- `nvsahipostprocess` — merges duplicate detections from overlapping slices (two-phase GreedyNMM).

The **YOLO26** (NMS-free) and **YOLOv9-C/GELAN** (EfficientNMS) detector families are pre-trained and
selectable at run time with `--model`. Full plugin reference: [`docs/PLUGINS.md`](docs/PLUGINS.md).

## Contents

- [Compatibility](#compatibility) · [Quick Start](#quick-start) · [**Models — pick one and run**](#models--pick-one-and-run)
- [Documentation](#documentation) · [Results Summary](#results-summary) · [Training](#training-not-in-this-repo) · [Limitations](#limitations) · [License](#license)

## Compatibility

| Component | DeepStream 8.0 | DeepStream 9.0 |
|-----------|----------------|----------------|
| DeepStream SDK | 8.0 | 9.0 |
| CUDA Toolkit | 12.8 | 13.1 |
| TensorRT | 10.9.0 | 10.14.1 |
| GStreamer | 1.24.2 | 1.24.2 |
| Python bindings | `pyds 1.2.2` | built from source |

`install.sh` detects the DeepStream version, builds the SAHI plugins, and builds
**`libnvds_infer_yolo.so`** — the custom parser **required** by the bundled ONNX models (they use
TensorRT EfficientNMS / NMS-free outputs the stock sample parser does not decode). Details:
[`docs/INSTALL.md`](docs/INSTALL.md).

## AI coding assistant support

The repository ships a [`CLAUDE.md`](CLAUDE.md) and [`.claude/skills/`](.claude/skills/) so **AI
coding assistants** (e.g. Claude Code) can work with this project out of the box — they pick up the
build/run workflow, the registered models, and the pipeline's non-obvious rules (`cluster-mode=2`,
`batch = tiles/frame`, the overload alert) automatically. If you use an AI code assistant, just open
it at the repository root; no extra setup is needed.

## Quick Start

This repository uses [Git LFS](https://git-lfs.com/) for ONNX model files.

```bash
git lfs install
git clone https://github.com/levipereira/deepstream-sahi.git
cd deepstream-sahi
```

Run a DeepStream container:

```bash
docker run -it --name deepstream-sahi --net=host --gpus all \
    -v `pwd`:/apps/deepstream-sahi \
    -w /apps/deepstream-sahi \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch
```

Inside the container:

```bash
/apps/deepstream-sahi/install.sh
source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate
cd /apps/deepstream-sahi/python_test/deepstream-test-sahi
python3 deepstream_test_sahi.py --model visdrone-full-640 --no-display --csv -i ../videos/aerial_crowding_01.mp4
```

Test videos are on [Google Drive](https://drive.google.com/drive/folders/1CRLnuH9AtTwmxRz7z-Mtu6ErKx__VMK4)
→ place them in `python_test/videos/`. Container variants, display notes, and rebuild mode:
[`docs/INSTALL.md`](docs/INSTALL.md).

## Models — pick one and run

Models are pre-trained and selected with `--model`. Each maps to a pgie + preprocess config; the
ONNX (Git LFS) and per-model training/accuracy provenance live in
[`model_zoo/visdrone_yolo26/`](model_zoo/visdrone_yolo26/). Full table + how to add a model:
[`docs/USAGE.md`](docs/USAGE.md).

| `--model` | Family | Input | Output → parser |
|-----------|--------|:-----:|-----------------|
| `visdrone-full-640` | YOLOv9-C (GELAN) | 640 | EfficientNMS → `NvDsInferYoloNMS` |
| `visdrone-sliced-448` | YOLOv9-C (GELAN) | 448 | EfficientNMS → `NvDsInferYoloNMS` |
| `visdrone-yolo26n-sliced-416` | YOLO26 (NMS-free) | 416 | `[N,6]` → `NvDsInferYoloE2E` |
| `visdrone-yolo26s-sliced-448` | YOLO26 (NMS-free) | 448 | `[N,6]` → `NvDsInferYoloE2E` |

```bash
# inside the container, pyds venv active, from python_test/deepstream-test-sahi
python3 deepstream_test_sahi.py --model visdrone-yolo26n-sliced-416 --no-display --csv \
    -i ../videos/aerial_crowding_01.mp4            # CSV = per-frame detections; PERF lines = FPS
python3 deepstream_test_sahi.py --model <model> --output-mp4 results/out.mp4 -i <video>   # annotated video
```

> **Tuning (important for FPS/accuracy):** set `batch-size` (pgie) = `network-input-shape[0]`
> (preprocess) = **tiles/frame** (41 @ slice 416, 29 @ slice 448, 16 @ slice 640 on a 2560×1440
> source) so each frame is one inference — roughly 3× FPS with no accuracy change. Slice size is the
> speed/recall knob (smaller = more recall, slower). **Multi-camera:** batch is `cameras × tiles/frame`,
> so aim for a **low tile count**. Always `cluster-mode=2`. See
> [`docs/SAHI_MODEL_BENCHMARK.md`](docs/SAHI_MODEL_BENCHMARK.md) → **Deployment planning**.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALL.md) | container setup, dependencies, plugin build |
| [Usage Guide](docs/USAGE.md) | pipeline execution, CLI arguments, adding a model |
| [Plugin Reference](docs/PLUGINS.md) | plugin properties, algorithms, tuning guide |
| [SAHI Model Benchmark](docs/SAHI_MODEL_BENCHMARK.md) | in-pipeline FPS/detection (YOLO26 vs GELAN), bottleneck analysis, batch/tile tuning |
| [Plugin Review](docs/PLUGIN_REVIEW.md) | nvsahipre/postprocess code review + optimizations |
| [Model Zoo](model_zoo/visdrone_yolo26/README.md) | training provenance, accuracy, TensorRT benchmarks |
| [Training Guide](docs/TRAINING.md) | training workflow for the sliced YOLOv9-C samples |
| [Test Results](docs/TEST_RESULTS.md) | full evaluation data and charts |
| [Parameter Tests](docs/PARAMETER_TESTS.md) · [Dense Crowd](docs/PARAMETER_TESTS_CROWDING.md) | postprocess parameter validation |

## Results Summary

Detection counts per frame, `2560×1440` input, FP16 — SAHI recovers small-object scale that a single
full-frame resize loses:

| Video | `full-640` no-SAHI → SAHI | `sliced-448` no-SAHI → SAHI |
|-------|:-------------------------:|:---------------------------:|
| `aerial_crowding_01` | 13.8 → 84.2 | 2.3 → 85.3 |
| `aerial_crowding_02` | 206.2 → 664.7 | 35.9 → 614.9 |
| `aerial_vehicles` | 92.3 → 252.5 | 28.6 → 226.7 |

### Pipeline FPS

Real, in-pipeline FPS (RTX 4090, FP16, `fakesink sync=false`, `2560×1440` source, slice = 416/41 tiles
for YOLO26n, 448/29 for GELAN). Setting `batch-size = tiles/frame` makes each frame one inference:

| Model | Input | Median FPS | With `batch = tiles/frame` |
|-------|:-----:|:----------:|:--------------------------:|
| **YOLO26n** | 416 | 135.6 | **389** (~3×) |
| YOLOv9-C / GELAN | 448 | 76.5 | — |

The postprocess merge is ~0.18 ms/frame — **not** the bottleneck; tiling and detection volume dominate.
Full benchmark: [`docs/SAHI_MODEL_BENCHMARK.md`](docs/SAHI_MODEL_BENCHMARK.md) ·
[`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md). All `nvsahipostprocess` parameters are validated by an
automated suite (21/21 across moderate and very-dense scenes): [`docs/PARAMETER_TESTS.md`](docs/PARAMETER_TESTS.md).

### Trained Models — Accuracy & TensorRT Throughput

Val accuracy (VisDrone sliced, 11 classes) and pure-GPU **inferences/second** (`img/s`) swept over
TensorRT batch size — FP16, RTX 4090, trtexec v10.14, single stream. `img/s` = inferences per second;
**bold** = peak. Provenance: [`model_zoo/visdrone_yolo26/`](model_zoo/visdrone_yolo26/).

| Model | Input | mAP<br>.50:.95 | mAP<br>.50 | b1 | b8 | b16 | b32 | b64 | b128 | b256 |
|-------|:-----:|:----:|:----:|----:|----:|----:|----:|----:|----:|----:|
| **YOLO26n** | 416 | 0.439 | 0.694 | 2,313 | 11,389 | 15,557 | 18,010 | **18,356** | 17,021 | 15,997 |
| YOLO26s | 448 | 0.368 | 0.649 | 1,858 | 6,576 | **7,604** | 7,555 | 7,057 | 6,767 | 6,508 |

Latency per batch (GPU-compute mean, ms — same runs as above). Per-image latency = batch latency ÷ batch,
so larger batches are far more efficient per image:

| Model | b1 | b8 | b16 | b32 | b64 | b128 | b256 |
|-------|----:|----:|----:|----:|----:|----:|----:|
| **YOLO26n** | 0.43 | 0.70 | 1.03 | 1.78 | 3.49 | 7.52 | 16.00 |
| YOLO26s | 0.54 | 1.22 | 2.10 | 4.23 | 9.07 | 18.92 | 39.33 |

Peak: **yolo26n 18,356 img/s @ batch 64**, **yolo26s 7,604 img/s @ batch 16**; latency/throughput knee
at batch 16 for both. Full sweep + per-image latency:
[`model_zoo/visdrone_yolo26/03_results/PERFORMANCE_TRT.md`](model_zoo/visdrone_yolo26/03_results/PERFORMANCE_TRT.md).

### Example Charts

<p align="center">
  <img src="test_results/comparison_aerial_crowding_01_visdrone-full-640-sahi_vs_aerial_crowding_01_visdrone-full-640-no-sahi/01_total_objects_over_frames.png" width="80%" alt="Total objects per frame for aerial_crowding_01"/>
  <img src="test_results/comparison_aerial_crowding_02_visdrone-full-640-sahi_vs_aerial_crowding_02_visdrone-full-640-no-sahi/01_total_objects_over_frames.png" width="80%" alt="Total objects per frame for aerial_crowding_02"/>
  <img src="test_results/comparison_aerial_vehicles_visdrone-full-640-sahi_vs_aerial_vehicles_visdrone-full-640-no-sahi/01_total_objects_over_frames.png" width="80%" alt="Total objects per frame for aerial_vehicles"/>
</p>

### Video Demos

| Dense Pedestrian Crowd | Very Dense Crowd | Dense Vehicle Traffic |
|:---:|:---:|:---:|
| [![Dense Pedestrian Crowd](https://img.youtube.com/vi/_W_wBDvpzzY/hqdefault.jpg)](https://www.youtube.com/watch?v=_W_wBDvpzzY&list=PLJMGcwo73q30LtZaCw1VQ7UGPvGsmVlco) | [![Very Dense Crowd](https://img.youtube.com/vi/RFX8hIWscgw/hqdefault.jpg)](https://www.youtube.com/watch?v=RFX8hIWscgw&list=PLJMGcwo73q33YfssoIGBIMu51EPJBobxf) | [![Dense Vehicle Traffic](https://img.youtube.com/vi/3CxEp90Jy60/hqdefault.jpg)](https://www.youtube.com/watch?v=3CxEp90Jy60&list=PLJMGcwo73q33HWQfjHUD_exEVOUSnlbsA) |

## Training (not in this repo)

This repository **deploys** pre-trained models; it does not train them. Reproduce/retrain with the
**original upstream repos** — **YOLO26** via official Ultralytics (`yolo detect train …`, then
`yolo export format=onnx dynamic=True simplify=True`) and **YOLOv9-C/GELAN** via the upstream YOLOv9
repo. The exact dataset (VisDrone sliced 416, 11 classes), commands, hyperparameters, accuracy and
TensorRT benchmarks are bundled under
[`model_zoo/visdrone_yolo26/`](model_zoo/visdrone_yolo26/); YOLOv9-C notes in
[`docs/TRAINING.md`](docs/TRAINING.md).

## Limitations

- **Object-count overload:** above ~2000 objects in a single frame, the OSD draw and GreedyNMM merge
  become the bottleneck and FPS drops sharply (the pipeline emits a one-time `[WARN]`). Bound it by
  raising `pre-cluster-threshold`, using fewer/larger tiles, or capping `max-detections`.
- **`cluster-mode=2` required.** `cluster-mode=4` renders wrong boxes in the OSD.
- Bidirectional NMM (transitive merge chains) is not implemented; GreedyNMM covers real-time use.
- Merged mask resolution is capped at 512×512. Multi-source runs use OpenMP parallelism but are not
  benchmarked end-to-end.

See [`docs/PLUGINS.md`](docs/PLUGINS.md) for the full property reference and algorithm details.

## License

This repository is **multi-licensed per component** — the SPDX header in each source file is
authoritative; see [`LICENSE`](LICENSE) at the root for the summary.

| Component | License |
|-----------|---------|
| **`nvsahipostprocess`** plugin (original work) | **[Apache-2.0](deepstream_source/gst-plugins/gst-nvsahipostprocess/LICENSE)** |
| **`nvsahipreprocess`** plugin (derivative of NVIDIA's `gst-nvdspreprocess` sample) + rest of the repo | **[NVIDIA DeepStream SDK EULA](https://developer.nvidia.com/deepstream-eula)** |
| `nvdsinfer_yolo` parser | see [its `LICENSE`](deepstream_source/libs/nvdsinfer_yolo/LICENSE) |

The Apache-2.0 license covers the postprocess plugin's **source**; building and running it still
requires the NVIDIA DeepStream SDK, which is governed by NVIDIA's agreement. Preserve all per-file
copyright/SPDX notices when redistributing.
