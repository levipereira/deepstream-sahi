# model_zoo

Trained-model packages developed for this project, with full training/benchmark provenance.

## Packages

### `visdrone_yolo26/` — VisDrone (sliced 416) YOLO26 detectors
Complete "Part 1" package: dataset construction, training, accuracy + TensorRT performance results,
trained weights, and ONNX (dynamic batch) for **Ultralytics YOLO26** (n / s) on VisDrone sliced into
416×416 tiles.

- Start at [`visdrone_yolo26/README.md`](visdrone_yolo26/README.md).
- Deployable artifacts: `visdrone_yolo26/04_models/*/` (`.onnx` + `.pt`, **Git LFS**).
- ONNX export params / DeepStream notes: `visdrone_yolo26/05_deployment/ONNX_EXPORT.md`.

YOLO26 is **end-to-end NMS-free** — its ONNX emits `[B, 300, 6]` (`x1,y1,x2,y2,score,label`),
decoded in DeepStream by `NvDsInferYoloE2E` with no extra clustering step. **YOLO26n** (mAP 0.439)
is the recommended deployment model; **YOLO26s** trades speed for a larger backbone.

## Git LFS
`*.onnx`, `*.pt` are tracked via Git LFS (see repo `.gitattributes`). Run
`git lfs install` before cloning/pulling to fetch the binaries.
