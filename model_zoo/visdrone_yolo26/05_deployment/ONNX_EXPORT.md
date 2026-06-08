# Deployment — ONNX export & TensorRT / DeepStream

This document records the exact ONNX export parameters used for YOLO26 models in the benchmark and
how to build TensorRT engines for DeepStream. All ONNX files are exported with **dynamic batch size**.

YOLO26 ONNX output is end-to-end NMS-free: the model head already emits final detections, so
DeepStream needs no NMS clustering downstream.

## 1. Exported files

| Model | ONNX file | Input `images` (NCHW) | Output `output0` | Dynamic axes | opset | Size |
|-------|-----------|----------------------|-----------------|--------------|:-----:|-----:|
| YOLO26n | `04_models/yolo26n/best.onnx` | `[batch,3,height,width]` | `[batch,300,6]` | batch, H, W | 17 | 10.4 MB |
| YOLO26s | `04_models/yolo26s/best.onnx` | `[batch,3,height,width]` | `[batch,300,6]` | batch, H, W | 17 | 38.6 MB |

`300` = maximum detections from the end-to-end head.

**Output column layout:** `x1, y1, x2, y2, score, label` (xyxy in pixels, confidence, class id).

## 2. YOLO26 export — exact command & parameters

Standard Ultralytics export. Install dependencies first if needed:
```bash
pip install onnx onnxslim onnxruntime
```

```bash
conda activate yolo26

# YOLO26n — imgsz 416 (matches training resolution)
yolo export model=best.pt format=onnx dynamic=True simplify=True opset=17 imgsz=416

# YOLO26s — imgsz 448 (matches training resolution)
yolo export model=best.pt format=onnx dynamic=True simplify=True opset=17 imgsz=448
```

Produces `best.onnx` next to the `.pt` file.

| Arg | Value | Meaning |
|-----|-------|---------|
| `format` | `onnx` | export target |
| `dynamic` | `True` | makes batch, height, and width axes dynamic |
| `simplify` | `True` | onnxslim graph simplification |
| `opset` | `17` | ONNX opset (optional; 17 is recommended) |
| `imgsz` | `416` (n) / `448` (s) | the `opt` spatial size baked as default |

The input tensor is named `images`; the output tensor is named `output0`.

> **End-to-end / NMS-free.** YOLO26's ONNX already outputs final detections `[B,300,6]`; no
> post-export NMS step is required. DeepStream should **not** cluster — use `cluster-mode=2`
> (see DeepStream notes below).
>
> `dynamic=True` also makes the spatial dims dynamic. If you need batch-only dynamic (fixed HxW,
> simpler TRT profile), re-export with a fixed `imgsz` and pin H=W in the trtexec profile
> (`min=opt=max` for H and W).

## 3. Build TensorRT engines (TRT >= 10.6 for FP16)

Dynamic shapes require an optimization profile (`min`/`opt`/`max`). Pin H=W to the trained
resolution for a straightforward profile.

```bash
# YOLO26n (416)
trtexec --onnx=04_models/yolo26n/best.onnx \
    --saveEngine=yolo26n.engine --fp16 \
    --minShapes=images:1x3x416x416 \
    --optShapes=images:8x3x416x416 \
    --maxShapes=images:16x3x416x416

# YOLO26s (448): substitute 416 -> 448 and the onnx path above
```

Adjust `min/opt/max` batch to match your DeepStream `batch-size`. For a single-stream pipeline use
`1x3xHxW` for all three profiles.

You can also let DeepStream build the engine on first run — place `best.onnx` next to the pgie
config and DeepStream will generate and cache `best.engine` automatically.

## 4. DeepStream nvinfer notes

| Property | Value |
|----------|-------|
| Input tensor name | `images` (NCHW) |
| Output tensor name | `output0` |
| Output shape | `[B, 300, 6]` — x1,y1,x2,y2,score,label (pixels) |
| NMS / clustering | `cluster-mode=2` (required; `cluster-mode=4` renders wrong OSD boxes) |
| `network-mode` | `2` (FP16) |
| `num-detected-classes` | `11` (VisDrone) |
| bbox parser | `NvDsInferYoloE2E` (`output-blob-names=output0`) |

The `NvDsInferYoloE2E` parser (`libnvds_infer_yolo.so`) reads the `[N,6]` end-to-end output
directly — no NMS inside the parser, only score thresholding.

## 5. Verification — dynamic batch confirmed

Both ONNX files were checked with onnxruntime across batch 1/2/4/8:

| Model | b1 | b2 | b4 | b8 | same-image batch vs b1 |
|-------|----|----|----|----|------------------------|
| YOLO26n | (1,300,6) | (2,300,6) | (4,300,6) | (8,300,6) | ✓ Δ ≈ 1.5e-7 |
| YOLO26s | (1,300,6) | (2,300,6) | (4,300,6) | (8,300,6) | ✓ |

("same-image batch vs b1": feeding the same image replicated B× reproduces the batch=1 result to
float epsilon.)

Quick re-check:
```bash
python - <<'PY'
import onnxruntime as ort, numpy as np
s = ort.InferenceSession("04_models/yolo26n/best.onnx",
                         providers=["CPUExecutionProvider"])
for b in (1, 2, 4, 8):
    out = s.run(None, {"images": np.zeros((b, 3, 416, 416), np.float32)})[0]
    print("batch", b, "-> output", out.shape)   # (b, 300, 6)
PY
```
