# VisDrone SAHI Project — Part 1: Training & Benchmarking (YOLO26)

Detection on **VisDrone** sliced into **416×416 tiles** (SAHI-style tiling) with **Ultralytics
YOLO26** (n / s). This directory is **Part 1**: dataset construction, model **training**, accuracy
**results**, and TensorRT **performance** benchmarking. It contains every config, script, log, weight,
and ONNX needed to reproduce the work and to build **Part 2** (the SAHI tiled-inference / deployment
pipeline) on top.

> Everything here is in English. Trained weights are provided as **`.pt`** plus **`.onnx`**
> (dynamic batch). No TensorRT `.engine` files are shipped — build them on the target host
> (see `05_deployment/ONNX_EXPORT.md`).

---

## Headline results

**Accuracy** (val, COCO mAP) — full breakdown in [`03_results/ACCURACY.md`](03_results/ACCURACY.md):

| Model | Params | Input | mAP@.50:.95 | mAP@.50 | Training status |
|-------|-------:|------:|------------:|--------:|-----------------|
| **YOLO26n** | ≈2.5 M | 416 | **0.439** | **0.694** | ✅ completed (80 ep, best == last, still rising) |
| YOLO26s     | ≈9.5 M | 448 | 0.368 | 0.649 | ✅ completed (45/100 logged) |

**Performance** (TensorRT **FP16**, RTX 4090, pure GPU compute) — full sweep in
[`03_results/PERFORMANCE_TRT.md`](03_results/PERFORMANCE_TRT.md):

| Model | Peak throughput | @batch | Latency @b1 | Best latency/throughput knee |
|-------|----------------:|:------:|------------:|------------------------------|
| yolo26n | 18,356 img/s | 64 | 0.43 ms | batch 16 |
| yolo26s | 7,604 img/s  | 16 | 0.54 ms | batch 16 |

**Bottom line:** **YOLO26n** is the recommended model — best accuracy in this bundle (mAP 0.439, and
still rising at the 80-epoch cutoff) while running at ~18 k img/s, sub-0.5 ms latency at batch 1.
**YOLO26s** has a larger backbone but logged lower mAP here (only 45/100 epochs). Multi-stream gives
no extra peak throughput (a single stream already saturates the GPU at batch ≥ 16).

---

## Directory layout

```
visdrone_yolo26/                    ← Part 1 root
├── README.md                       ← this index
│
├── 01_dataset/                     ← how the sliced dataset was built
│   ├── DATASET.md                    spec, on-disk composition, Google Drive distribution
│   ├── pipeline/                     exact scripts that built the dataset
│   │   ├── 01_create_visdrone_yolo.py    (raw VisDrone DET+VID → YOLO)
│   │   └── 02_create_visdrone_slice.py   (YOLO → 416 tiles, 20% overlap)
│   ├── visdrone_slice.yaml           Ultralytics data config (sliced)
│   └── roboflow_upload.py            optional Roboflow upload (reads $ROBOFLOW_API_KEY)
│
├── 02_training/                    ← how training was done (Part 1 core)
│   ├── TRAINING.md                   full experiment log: params, commands, outcomes
│   └── yolo26/                       yolo26{n,s}_args.yaml (exact YOLO hyperparameters)
│
├── 03_results/                     ← accuracy + performance
│   ├── ACCURACY.md                   COCO AP breakdown, per-class context
│   ├── PERFORMANCE_TRT.md            TRT FP16 throughput/latency (batch sweep + multi-stream)
│   ├── accuracy_summary.csv          machine-readable accuracy summary
│   ├── trt_pass1_batch_sweep.csv     raw TRT pass-1 (batch 1..256, single stream)
│   ├── trt_pass2_multistream.csv     raw TRT pass-2 (streams 1..8)
│   ├── benchmark_scripts/            run_bench*.sh + bench_worker*.sh (reproduce the TRT bench)
│   └── raw_trtexec_logs/             trtexec logs (pass1/ + pass2/)
│
├── 04_models/                      ← trained weights + ONNX (no engines)
│   ├── yolo26n/       best.pt  best.onnx  args.yaml  results.csv
│   └── yolo26s/       best.pt  best.onnx  args.yaml  results.csv
│
└── 05_deployment/                  ← export for inference (input to Part 2)
    └── ONNX_EXPORT.md                ONNX export params (dynamic batch) + TRT/DeepStream notes
```

---

## Where to look

| You want to… | Go to |
|--------------|-------|
| Rebuild the sliced dataset from raw VisDrone | [`01_dataset/DATASET.md`](01_dataset/DATASET.md) §4 |
| See exact training commands & hyperparameters | [`02_training/TRAINING.md`](02_training/TRAINING.md) |
| Compare accuracy across models | [`03_results/ACCURACY.md`](03_results/ACCURACY.md) |
| Compare throughput/latency (TensorRT) | [`03_results/PERFORMANCE_TRT.md`](03_results/PERFORMANCE_TRT.md) |
| Get weights / ONNX | [`04_models/`](04_models/) |
| Export / deploy the models | [`05_deployment/ONNX_EXPORT.md`](05_deployment/ONNX_EXPORT.md) |

## Model files at a glance

- **YOLO26** ships `best.pt` + `best.onnx` (output `[B, 300, 6]` = `x1,y1,x2,y2,score,label`,
  **dynamic batch**, end-to-end **NMS-free** → no clustering needed; decoded by `NvDsInferYoloE2E`).
- All ONNX verified for dynamic batch (1/2/4/8); export parameters in `05_deployment/ONNX_EXPORT.md`.

## Reproduction environments
- YOLO26 training/export: conda `yolo26` (ultralytics 8.4.33).
- TensorRT benchmark: docker `nvcr.io/nvidia/tensorrt:25.11-py3` (trtexec v10.14).

## Continuing to Part 2 (SAHI)
Part 2 (tiled SAHI inference / deployment) builds on `04_models/` (ONNX) and
`05_deployment/ONNX_EXPORT.md`. The recommended starting model is **YOLO26n** (best accuracy / speed).
Use the slicing parameters in `01_dataset/DATASET.md` (416×416, 20% overlap) to keep SAHI inference
consistent with training.

## Citation
If you use this benchmark, cite **Ultralytics YOLO** and **VisDrone2019**.
