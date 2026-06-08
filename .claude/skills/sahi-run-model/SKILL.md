---
name: sahi-run-model
description: Run / benchmark a DeepStream-SAHI model the user picks (FPS, per-frame detections, optional annotated MP4). Use when asked to "test a model", "run the SAHI pipeline", "benchmark <model>", or compare models.
---

# Run a SAHI model

The user picks a model by id (see the `MODELS` dict in
`python_test/deepstream-test-sahi/pipeline_common.py`, or run with a bad `--model` to list them).
Everything runs **inside the DeepStream container**.

## 1. Container + venv
```bash
# one persistent container, reused across runs (avoid rebuilding pyds each time):
docker run -d --name ds-sahi --gpus all --net=host \
    -v `pwd`:/apps/deepstream-sahi -w /apps/deepstream-sahi \
    --entrypoint sleep nvcr.io/nvidia/deepstream:9.0-triton-multiarch infinity
docker exec ds-sahi bash -lc 'cd /apps/deepstream-sahi && ./install.sh'   # first time only (slow: builds pyds)
```
Activate pyds in every exec: `source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate`.

## 2. Run (CSV = per-frame detection counts)
```bash
docker exec ds-sahi bash -lc '
  source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate
  cd /apps/deepstream-sahi/python_test/deepstream-test-sahi
  CUDA_VISIBLE_DEVICES=0 python3 deepstream_test_sahi.py --model <MODEL> --no-display --csv \
    -i ../videos/<video>.mp4'
```
- First run builds the TensorRT engine (~3 min), cached afterwards next to the ONNX.
- Annotated video for visual inspection: add `--output-mp4 results/out.mp4` (writes to `results/`,
  which is git-ignored and may be root-owned).
- SAHI knobs: `--slice-width/--slice-height` (tile size), `--overlap-w/--overlap-h`, `--no-full-frame`,
  `--match-metric 0|1` (IoU/IoS), `--match-threshold`.

## 3. Measure FPS + detections
- **FPS** = median of the `**PERF: {'stream0': X}` lines (printed every ~5 s); drop the first `0.0`
  and the warm-up window. Use a long enough video so several windows accumulate.
- **Detections** = the `results/*.csv` (`total_objects` column) → mean/frame, total.
- **Postprocess ms** = `GST_DEBUG=nvsahipostprocess:4` → `avg X ms/frame` lines.

## 4. Make it fast and correct
- Set `batch-size` (pgie) = `network-input-shape[0]` (preprocess) = **tiles/frame** (41 @ slice 416 /
  2560×1440) → 1 inference/frame.
- `cluster-mode=2` always.
- Object overload: above ~2000 objects/frame FPS collapses. Bound with `pre-cluster-threshold`, fewer tiles, or `max-detections` on `nvsahipostprocess`.

## 5. Two-GPU parallel benchmark
Run two models/videos at once, one per GPU:
```bash
docker exec ds-sahi bash -lc '... CUDA_VISIBLE_DEVICES=0 python3 ... &  CUDA_VISIBLE_DEVICES=1 python3 ... & wait'
```
Note: heavy CPU contention (decode/OSD) can skew parallel FPS — confirm a close number solo.
