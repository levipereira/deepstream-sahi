---
name: sahi-build-plugins
description: Rebuild and test the SAHI C++ plugins / nvinfer parser after editing them. Use when changing gst-nvsahipreprocess, gst-nvsahipostprocess, or libs/nvdsinfer_yolo, and you need to verify it compiles and the result is correct.
---

# Build & test the SAHI plugins

There is **no host build** — the plugins need the DeepStream SDK headers, so build inside the container.

## Build (fast path)
```bash
docker exec ds-sahi bash -lc 'cd /apps/deepstream-sahi && ./install.sh --plugins-only'
```
- `--plugins-only` rebuilds only `nvsahipreprocess`, `nvsahipostprocess`, and `libnvds_infer_yolo.so`
  (≈1–2 min). Flags `-Wall -Werror -O2`, so warnings fail the build.
- Outputs: `/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_sahi{pre,post}process.so`,
  `/opt/nvidia/deepstream/deepstream/lib/libnvds_infer_yolo.so`.
- Verify a parser symbol is exported: `nm -D .../libnvds_infer_yolo.so | grep NvDsInferYolo`.

## Correctness A/B (MANDATORY for postprocess/parser changes)

A perf change must NOT change detections. Compare obj/frame before vs after on a dense video:
```bash
# baseline (current committed version):
docker exec ds-sahi bash -lc '... python3 deepstream_test_sahi.py --model <m> --no-display --csv -i ../videos/aerial_crowding_01.mp4'
# -> read results/*.csv mean total_objects, then edit, rebuild --plugins-only, re-run, and the mean MUST match bit-for-bit.
```
To revert for the A/B: `git stash push -- <file>` → rebuild → run → `git stash pop` → rebuild.

A real example from this repo: the `SahiSpatialGrid::query` visited-stamp change was neutral (~23%
faster postprocess, identical results); a companion `cell_size`=median change altered results and was
**rejected**. Always isolate one change at a time when results move.

## Profiling
- Postprocess time: `GST_DEBUG=nvsahipostprocess:4` → `avg X ms/frame`.
- The GST latency tracer does **not** give per-element times for DeepStream elements; instrument with
  a 1 s counter like the one in `gstnvsahipostprocess.cpp` if you need preprocess timing.

## Gotchas
- `results/` and built `.so`/`.engine` are root-owned (container) — `sudo chown` on the host if needed.
- Keep the container alive between edits so the engine cache + pyds build are reused.
