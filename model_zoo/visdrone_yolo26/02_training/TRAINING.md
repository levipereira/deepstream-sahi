# Training log — YOLO26 on sliced VisDrone

This is the complete, self-contained record of YOLO26 training in this benchmark: exact commands,
hyperparameters, hardware, runtimes, and outcomes. Any number quoted in
[ACCURACY.md](../03_results/ACCURACY.md) can be traced back to the configs and logs here.

- Dataset build + slicing scripts: [DATASET.md](../01_dataset/DATASET.md)
- Headline numbers + caveats: [ACCURACY.md](../03_results/ACCURACY.md)
- Reproduction quick-start: [README.md](../README.md)

---

## 0. Hardware & environment

| Item | Value |
|------|-------|
| GPUs | 2 × NVIDIA RTX 4090 (24 GB each) |
| Host | `omninfsrv01`, Linux 5.15 |
| YOLO env | conda `yolo26`, Ultralytics 8.4.33, Python 3.11, AMP enabled |

---

## 1. Dataset

Training data is the **sliced VisDrone dataset** — 11 classes, tiles cut at 416×416 with 20% overlap
from the original VisDrone images. See [DATASET.md](../01_dataset/DATASET.md) for the full pipeline.
The Ultralytics data config is at `01_dataset/visdrone_slice.yaml`.

---

## 2. Experiment matrix

| Model | imgsz | Target epochs | Epochs run | val mAP@.50:.95 | val mAP@.50 | Params | Weights |
|-------|------:|:-------------:|:----------:|----------------:|------------:|-------:|---------|
| YOLO26n | 416 | 80 | 80 | **0.4391** | **0.6939** | ~2.5 M | `best.pt` |
| YOLO26s | 448 | 100 | 45 | **0.3679** | **0.6494** | ~9.5 M | `best.pt` |

YOLO26n reached its target epoch count; best equals last epoch (validation mAP was still rising).
YOLO26s was cut at 45 epochs — the 0.3679 figure is therefore a lower bound.

---

## 3. Training commands

Both models are trained with the Ultralytics CLI. Canonical form:

```bash
# YOLO26n — 2 GPUs, imgsz 416, 80 epochs
yolo detect train \
    data=visdrone_slice.yaml \
    model=yolo26n.pt \
    epochs=80 \
    imgsz=416 \
    batch=128 \
    device=0,1

# YOLO26s — 2 GPUs, imgsz 448, 100 epochs
yolo detect train \
    data=visdrone_slice.yaml \
    model=yolo26s.pt \
    epochs=100 \
    imgsz=448 \
    batch=64 \
    device=0,1
```

Full hyperparameter dumps (as saved by Ultralytics) are in `02_training/yolo26/yolo26n_args.yaml` and
`02_training/yolo26/yolo26s_args.yaml`.

---

## 4. Hyperparameters

### 4.1 YOLO26n

Key settings extracted from `yolo26n_args.yaml`:

| Parameter | Value |
|-----------|-------|
| `epochs` | 80 |
| `imgsz` | 416 |
| `batch` | 128 |
| `device` | 0,1 |
| `patience` | 20 |
| `optimizer` | auto |
| `pretrained` | true |
| `seed` | 0 |
| `amp` | true |
| `lr0` | 0.01 |
| `lrf` | 0.01 |
| `momentum` | 0.937 |
| `weight_decay` | 0.0005 |
| `warmup_epochs` | 3.0 |
| `cos_lr` | false |
| `close_mosaic` | 10 |
| `mosaic` | 0.0 |
| `iou` | 0.7 |
| `max_det` | 300 |

### 4.2 YOLO26s

Key settings extracted from `yolo26s_args.yaml`:

| Parameter | Value |
|-----------|-------|
| `epochs` | 100 |
| `imgsz` | 448 |
| `batch` | 64 |
| `device` | 0,1 |
| `patience` | 50 |
| `optimizer` | auto |
| `pretrained` | true |
| `seed` | 0 |
| `amp` | true |
| `lr0` | 0.01 |
| `lrf` | 0.01 |
| `momentum` | 0.937 |
| `weight_decay` | 0.0005 |
| `warmup_epochs` | 3.0 |
| `warmup_bias_lr` | 0.1 |
| `cos_lr` | true |
| `close_mosaic` | 0 |
| `mosaic` | 0.0 |
| `iou` | 0.7 |
| `max_det` | 300 |

---

## 5. Outcomes

### 5.1 YOLO26n

- Final result: **mAP@.50:.95 = 0.4391 / mAP@.50 = 0.6939** at epoch 80 (best == last epoch).
- Validation mAP was still rising at the end of training; more epochs would likely improve results.
- Trained with `pretrained=true` (initialized from the official YOLO26n COCO weights).
- Approximately 2.5 M parameters. `best.pt` shipped in `04_models/yolo26n/`.

### 5.2 YOLO26s

- Logged result (45 / 100 epochs): **mAP@.50:.95 = 0.3679 / mAP@.50 = 0.6494**.
- Training was not run to completion; reported numbers are a lower bound.
- Larger input (448) and model (~9.5 M params) compared to YOLO26n, but fewer epochs mean the
  advantage has not materialised yet.
- `best.pt` shipped in `04_models/yolo26s/`.

---

## 6. Reproduction notes

- Export to ONNX after training: `yolo export model=best.pt format=onnx dynamic=True simplify=True`.
- The visdrone_slice.yaml data path in the saved `args.yaml` is absolute (`/dataset/dfine/dataset/`);
  update it if reproducing on a different host.
- YOLO26s used `cos_lr=true` and `deterministic=true` (reproducible but slightly slower) while YOLO26n
  used the default cosine schedule disabled and `deterministic=false`.
- Both runs used `mosaic=0.0` and `auto_augment=randaugment` (YOLO26n) / `auto_augment=false` (YOLO26s).

---

## 7. Provenance / file map

| Artifact | Path in bundle |
|----------|----------------|
| YOLO data config | `01_dataset/visdrone_slice.yaml` |
| YOLO26n hyperparameters | `02_training/yolo26/yolo26n_args.yaml` |
| YOLO26s hyperparameters | `02_training/yolo26/yolo26s_args.yaml` |
| Dataset build scripts | `01_dataset/pipeline/01..03_*.py` |
| Trained weights | `04_models/yolo26n/best.pt`, `04_models/yolo26s/best.pt` |
| Training metrics | `04_models/yolo26{n,s}/results.csv` |
| Machine-readable summary | `results.csv` |

Every AP value in ACCURACY.md is the best row of the corresponding `results.csv`.
