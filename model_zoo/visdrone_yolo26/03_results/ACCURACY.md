# Results — VisDrone YOLO26 (sliced 416×416 tiles)

All models were trained and evaluated on the **same sliced VisDrone dataset** (416×416 tiles,
20% overlap, 11 classes). See [DATASET.md](../01_dataset/DATASET.md) for the dataset specification
and the exact slicing/conversion scripts.

## Headline results (val, COCO mAP)

| Model | Params | Framework | Input | mAP@.50:.95 | mAP@.50 | Best / logged epochs | Status |
|-------|-------:|-----------|------:|------------:|--------:|:--------------------:|:------:|
| **YOLO26n** | ≈ 2.5 M | Ultralytics | 416 | **0.4391** | **0.6939** | 80 / 80 | done (best = last, still rising) |
| YOLO26s     | ≈ 9.5 M | Ultralytics | 448 | 0.3679 | 0.6494 | 45 / 100 | done (45 of 100 logged) |

The machine-readable summary lives at `03_results/accuracy_summary.csv`.

### Reading the results

- **YOLO26n is the recommended model** in this bundle. It converges to a higher mAP (0.439 vs 0.368)
  at a lower parameter count (≈ 2.5 M vs ≈ 9.5 M) and its training run ran to completion.
- **YOLO26s shows a lower mAP** because only 45 of the planned 100 epochs were logged. The curve had
  not yet plateaued (mAP@.50:.95 was still climbing at epoch 45), so 0.368 is a lower bound on what
  the full run would achieve. It was also trained at a slightly larger input (448 vs 416).
- The Ultralytics validator reports `mAP@.50:.95` and `mAP@.50` directly. Per-object-size breakdown
  (AP_small / AP_medium / AP_large) and AR@100 are not produced by the standard Ultralytics val log
  and are therefore not available for these runs (NA).

## YOLO val log summary

| Model | mAP@.50:.95 | mAP@.50 | Precision | Recall | Epochs logged |
|-------|------------:|--------:|----------:|-------:|--------------:|
| YOLO26n | 0.4391 | 0.6939 | 0.821 | 0.605 | 80 |
| YOLO26s | 0.3679 | 0.6494 | 0.797 | 0.570 | 45 |

## Dataset and eval context

- **Dataset:** VisDrone sliced at 416×416 with 20% overlap, 11 classes (pedestrian, people, bicycle,
  car, van, truck, tricycle, awning-tricycle, bus, motor, others). See `01_dataset/DATASET.md`.
- VisDrone is heavily dominated by small objects, so `mAP@.50:.95` reflects small-object detection
  quality more than most benchmarks.
- Eval was run with the Ultralytics validator on the full sliced val split.

## Provenance (where each number comes from)

| Model | Source of metric | Trained weights in bundle |
|-------|------------------|---------------------------|
| YOLO26n | `04_models/yolo26n/results.csv` (epoch 80) | `best.pt` |
| YOLO26s | `04_models/yolo26s/results.csv` (epoch 45) | `best.pt` |

## Excluded run

An older `outputs/yolo11n_visdrone` run (mAP ≈ 0.19) was **excluded**: its `args.yaml` points to a
different data yaml (`visdrone_yolo.yaml`, i.e. the **non-sliced** full-frame dataset
`/dataset/dfine/dataset/visdrone_yolo/`) and the directory naming is inconsistent with its
recorded model. It is **not** comparable to the sliced benchmark and would be misleading here.
