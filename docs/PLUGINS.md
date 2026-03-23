# Plugin Reference

Complete documentation for the two GStreamer plugins provided by this project.

---

## nvsahipreprocess

**SAHI dynamic-slice pre-processor for DeepStream.**

Replaces static ROI groups used by NVIDIA's `nvdspreprocess` with dynamic per-frame SAHI slice computation. For each incoming frame, the plugin divides the image into overlapping tiles (slices), crops and scales each tile to the network input resolution via `NvBufSurfTransform`, then delegates tensor preparation to the same custom library interface used by `nvdspreprocess`. The resulting `GstNvDsPreProcessBatchMeta` is consumed by `nvinfer` with `input-tensor-meta=1`.

### Pipeline Position

```
nvstreammux → nvsahipreprocess → nvinfer (input-tensor-meta=1) → queue → nvsahipostprocess → ...
```

### Pad Templates

| Pad  | Direction | Availability | Caps |
|------|-----------|-------------|------|
| sink | Sink      | Always       | `video/x-raw(memory:NVMM), format={NV12, RGBA, I420}` |
| src  | Source    | Always       | `video/x-raw(memory:NVMM), format={NV12, RGBA, I420}` |

### GStreamer Element Properties

These properties are set directly on the element in the pipeline (e.g. via Python `set_property()` or `gst-launch` command line).

| Property | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `unique-id` | uint | 15 | 0 – UINT_MAX | Unique identifier for this element instance. Used to match with `target-unique-ids` in downstream elements. |
| `enable` | boolean | true | — | Enable the plugin. When `false`, the element operates in passthrough mode (buffers flow through untouched). |
| `gpu-id` | uint | 0 | 0 – UINT_MAX | GPU device ID to use for scaling and tensor operations. |
| `config-file` | string | `""` | — | Path to the configuration file for tensor preparation parameters (see [Configuration File](#configuration-file) below). **Required.** |
| `slice-width` | uint | 640 | 1 – UINT_MAX | Width of each SAHI slice in pixels. |
| `slice-height` | uint | 640 | 1 – UINT_MAX | Height of each SAHI slice in pixels. |
| `overlap-width-ratio` | float | 0.2 | 0.0 – 0.99 | Horizontal overlap between adjacent slices, expressed as a fraction of `slice-width`. |
| `overlap-height-ratio` | float | 0.2 | 0.0 – 0.99 | Vertical overlap between adjacent slices, expressed as a fraction of `slice-height`. |
| `enable-full-frame` | boolean | true | — | Append the entire frame as an additional slice. This ensures large objects spanning multiple slices are still detected — standard SAHI behaviour. Disable only for debugging. |
| `target-unique-ids` | string | `""` | — | Semicolon-separated list of downstream GIE `unique-id` values for which tensors are prepared (e.g. `"1"` or `"3;4;5"`). |

### Configuration File

The config file uses GLib key-file (INI) format and contains a mandatory `[property]` section and an optional `[user-configs]` section. Unlike `nvdspreprocess`, there are **no `[group-N]` / ROI sections** — slices are computed dynamically from the element properties above.

#### `[property]` Section — Required Keys

| Key | Type | Description |
|-----|------|-------------|
| `enable` | int (0/1) | Enable or disable the plugin. |
| `target-unique-ids` | int list (`;`-separated) | GIE unique-ids to target. |
| `processing-width` | int | Width to scale each slice to before tensor preparation. Must match the network input width. |
| `processing-height` | int | Height to scale each slice to before tensor preparation. Must match the network input height. |
| `maintain-aspect-ratio` | int (0/1) | Maintain aspect ratio during scaling, padding with black. |
| `symmetric-padding` | int (0/1) | When `maintain-aspect-ratio=1`, pad symmetrically (center the image) instead of padding only right/bottom. |
| `network-input-order` | int | Tensor layout: `0` = NCHW, `1` = NHWC. |
| `network-input-shape` | int list (`;`-separated) | Full tensor shape. For NCHW: `B;C;H;W` (e.g. `16;3;640;640`). The batch dimension (`B`) determines the maximum number of slices processed in a single GPU batch. |
| `network-color-format` | int | Color format: `0` = RGB, `1` = BGR, `2` = GRAY. |
| `tensor-data-type` | int | Data type: `0` = FP32, `1` = UINT8, `2` = INT8, `3` = UINT32, `4` = INT32, `5` = FP16. |
| `tensor-name` | string | Name of the input tensor (must match the model's input layer name, e.g. `images`). |
| `scaling-filter` | int | Interpolation filter: `0` = Nearest, `1` = Bilinear. |
| `scaling-pool-memory-type` | int | Surface memory type: `0` = Default, `1` = CUDA Pinned, `2` = CUDA Device, `3` = CUDA Unified, `4` = Surface Array. |
| `scaling-pool-compute-hw` | int | Compute backend for scaling: `0` = Default, `1` = GPU, `2` = VIC (Jetson only). |
| `custom-lib-path` | string | Absolute path to the shared library for tensor preparation (e.g. `/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so`). |
| `custom-tensor-preparation-function` | string | Name of the exported function in the custom library (e.g. `CustomTensorPreparation`). |

#### `[property]` Section — Optional Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `unique-id` | int | 15 | Plugin unique-id (can also be set via element property). |
| `gpu-id` | int | 0 | GPU device ID (can also be set via element property). |
| `scaling-buf-pool-size` | int | 6 | Number of buffers in the scaling surface pool. Increase if the pipeline stalls waiting for buffers. |
| `tensor-buf-pool-size` | int | 6 | Number of buffers in the tensor pool. |

#### `[user-configs]` Section

Arbitrary key-value pairs passed to the custom library's `initLib()` function as a `std::unordered_map<std::string, std::string>`.

| Key | Example Value | Description |
|-----|--------------|-------------|
| `pixel-normalization-factor` | `0.003921568` | Per-pixel normalization (1/255). Used by the default `libcustom2d_preprocess.so`. |

#### Example Config File

```ini
[property]
enable=1
target-unique-ids=1
network-input-order=0
maintain-aspect-ratio=1
symmetric-padding=1
processing-width=640
processing-height=640
scaling-buf-pool-size=6
tensor-buf-pool-size=6
network-input-shape=16;3;640;640
network-color-format=0
tensor-data-type=0
tensor-name=images
scaling-pool-memory-type=0
scaling-pool-compute-hw=0
scaling-filter=0
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so
custom-tensor-preparation-function=CustomTensorPreparation

[user-configs]
pixel-normalization-factor=0.003921568
```

### Slice Computation Algorithm

The slicing algorithm is ported from SAHI's Python `slicing.py:get_slice_bboxes()`:

1. Compute step sizes: `w_step = slice_width - (slice_width × overlap_width_ratio)`, same for height.
2. Iterate over the frame in a grid with those steps.
3. At each position, create a crop rectangle of `slice_width × slice_height`.
4. When a slice would extend past the frame boundary, shift it back so it ends exactly at the edge (ensures full coverage without padding).
5. If `enable-full-frame=true`, append one additional slice covering the entire frame (0, 0, frame_width, frame_height).

For a 1920×1080 frame with 640×640 slices and 0.2 overlap:
- Horizontal step = 640 − 128 = 512 → positions at x = 0, 512, 1024, 1280
- Vertical step = 640 − 128 = 512 → positions at y = 0, 440
- **Result**: 4×2 = 8 slices + 1 full-frame = **9 total** inference regions per frame.

### Architecture

The plugin uses a two-thread architecture (inherited from `nvdspreprocess`):

1. **Submission thread** (`submit_input_buffer`): computes slices, crops and scales each slice via batched `NvBufSurfTransformAsync`, then enqueues the batch.
2. **Output thread** (`gst_nvsahipreprocess_output_loop`): dequeues batches, waits for async transforms to complete, calls the custom tensor preparation function, and attaches `GstNvDsPreProcessBatchMeta` to the buffer before pushing downstream.

### Metadata Output

The plugin attaches `GstNvDsPreProcessBatchMeta` (meta type `NVDS_PREPROCESS_BATCH_META`) to each buffer. This metadata contains:

- `roi_vector` — one `NvDsRoiMeta` per slice with the original crop coordinates, scale ratios, and offsets.
- `tensor_meta` — pointer to the prepared GPU tensor buffer, shape, data type, and tensor name.
- `target_unique_ids` — the GIE IDs this tensor is intended for.

When `nvinfer` has `input-tensor-meta=1`, it reads this metadata directly instead of performing its own scaling, enabling zero-copy tensor handoff.

---

## nvsahipostprocess

**GreedyNMM post-processor for DeepStream (v1.2).**

Merges duplicate detections produced by sliced inference. Objects at slice boundaries appear in multiple overlapping slices, producing near-duplicate bounding boxes. This plugin applies an improved GreedyNMM algorithm to suppress or merge those duplicates. It operates entirely on `NvDsObjectMeta` — no tensor access, no CUDA kernels.

**v1.2 improvements**: two-phase merge algorithm, spatial hash grid indexing (O(n·k) vs O(n²)), per-class partitioning, instance-segmentation mask merge, cross-class label correction, configurable merge strategies, max-detections cap, per-frame debug statistics, and OpenMP parallel frame processing.

### Pipeline Position

```
nvinfer → queue → nvsahipostprocess → nvtracker → nvdsosd → sink
```

### Pad Templates

| Pad  | Direction | Availability | Caps |
|------|-----------|-------------|------|
| sink | Sink      | Always       | `video/x-raw(memory:NVMM), format={NV12, RGBA, I420}` |
| src  | Source    | Always       | `video/x-raw(memory:NVMM), format={NV12, RGBA, I420}` |

### GStreamer Element Properties

| Property | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `gie-ids` | string | `"-1"` | — | Semicolon-separated list of GIE `unique-component-id` values to process. `"-1"` or empty = all GIEs. Example: `"1;3;5"`. |
| `match-metric` | uint | 1 (IoS) | 0 – 1 | Overlap metric: `0` = **IoU** (Intersection over Union), `1` = **IoS** (Intersection over Smaller). IoS is recommended for sliced inference because objects near slice boundaries often have very different areas. |
| `match-threshold` | float | 0.5 | 0.0 – 1.0 | Overlap value above which two detections are considered duplicates. |
| `class-agnostic` | boolean | false | — | If `true`, compare detections across different class IDs. If `false`, detections are partitioned by class before NMM (much faster with many classes). |
| `enable-merge` | boolean | true | — | If `true`, merge suppressed boxes into survivors (GreedyNMM). If `false`, simply remove duplicates (standard NMS). |
| `two-phase-nmm` | boolean | true | — | Use two-phase GreedyNMM: phase 1 selects candidates using original (immutable) bboxes; phase 2 re-checks against the expanding bbox and only merges if still above threshold. Prevents aggressive chain-merging of non-overlapping detections. Set `false` for single-phase (more aggressive, v1.0 behavior). |
| `merge-strategy` | uint | 0 | 0 – 2 | Bbox merge method: `0` = **union** (min/max corners), `1` = **weighted** (score-weighted average), `2` = **largest** (keep larger bbox). |
| `max-detections` | int | -1 | -1 – INT_MAX | Maximum surviving detections per frame after merge. `-1` = unlimited. When exceeded, lowest-scoring survivors are removed. |
| `drop-mask-on-merge` | boolean | false | — | If `true`, clear segmentation masks when a merge occurs. If `false`, masks are composited via element-wise maximum in the merged bbox coordinate space. |

### GreedyNMM Algorithm (v1.2 — Two-Phase)

The algorithm processes each frame independently. When `class-agnostic=false`, detections are partitioned by class and NMM runs per-class (avoiding ~90% wasted cross-class comparisons).

**Phase 1 — Candidate selection (original coordinates):**

1. **Collect** all `NvDsObjectMeta` from the frame (optionally filtering by `gie-id`).
2. **Sort** by confidence (descending), with deterministic tie-breaking on bbox coordinates.
3. **Build spatial index** — a 2D hash grid with cell size equal to the largest detection dimension.
4. **Iterate** in score order. For each survivor `i`, query the spatial grid for nearby detections. For each candidate `j` (lower score):
   - Compute overlap using the configured metric against **original** (immutable) coordinates.
   - If overlap ≥ threshold: mark `j` as suppressed, add to `i`'s merge list.

**Phase 2 — Re-check and merge (expanding coordinates):**

5. For each survivor `i` with a merge list, iterate its candidates:
   - Re-check overlap against the **current** (expanding) bbox of `i`.
   - If still ≥ threshold: merge `j`'s bbox and mask into `i`.
6. **Apply results**: update metadata (bbox, confidence, class label, mask), remove suppressed objects.

This two-phase approach prevents cascade merging: if A absorbs B and expands, the expanded A does not absorb C unless C was already a candidate of original-A.

### Instance-Segmentation Mask Merge

When `enable-merge=true` and `drop-mask-on-merge=false`:

- If both detections carry `NvOSD_MaskParams` data, both masks are projected into the union bbox coordinate space using nearest-neighbor resampling, and the element-wise maximum is taken. The merged mask is written back to the surviving `NvDsObjectMeta`.
- If only one detection has a mask, it is resized to the merged bbox.
- If neither has a mask, no mask processing occurs (current behavior).

When `drop-mask-on-merge=true`, the mask is explicitly cleared on any merge. This is useful when exact mask merge is not needed and clean metadata is preferred.

### Match Metrics Explained

#### IoU (Intersection over Union)

```
IoU = intersection_area / (area_A + area_B - intersection_area)
```

Standard NMS metric. Works well when both boxes have similar sizes.

#### IoS (Intersection over Smaller) — Recommended

```
IoS = intersection_area / min(area_A, area_B)
```

Measures how much of the smaller box is contained within the larger one. Handles the common scenario where the full-frame detection is much larger than the slice detection of the same object.

### Debug Statistics

Per-frame merge statistics are emitted at the `LOG` level (6). See [Debug & Latency Profiling](#debug--latency-profiling) above for usage.

### Debug & Latency Profiling

Uses the standard GStreamer debug system via `GST_DEBUG`:

| Level | Command | Output |
|-------|---------|--------|
| INFO (4) | `GST_DEBUG=nvsahipostprocess:4` | PERF latency summary every ~1s |
| DEBUG (5) | `GST_DEBUG=nvsahipostprocess:5` | + element init, config, transform_ip per buffer |
| LOG (6) | `GST_DEBUG=nvsahipostprocess:6` | + per-frame NMM detail (dets, suppressed, merged, surviving) |

Example — latency profiling only:

```bash
GST_DEBUG=nvsahipostprocess:4 python3 deepstream_test_sahi.py --model visdrone-full-640 --no-display -i video.mp4
```

```
INFO  nvsahipostprocess ... PERF 1.0s: 30 batches, 30 frames | avg 0.28 ms/batch, 0.28 ms/frame | total 8.5 ms
```

Example — full per-frame detail:

```bash
GST_DEBUG=nvsahipostprocess:6 gst-launch-1.0 ...
```

```
LOG   nvsahipostprocess ... frame 42: collected 450 dets (gie_filter_all=1)
LOG   nvsahipostprocess ... frame 42: grid built 1920x1080, 450 rects
LOG   nvsahipostprocess ... frame 42: 450 dets, 200 suppressed, 120 merged, 250 surviving
```

### Performance

| Feature | Impact |
|---------|--------|
| Spatial hash grid | 10–50× fewer pair checks for spatially distributed detections |
| Per-class partitioning | Up to 10× for 10+ classes (class-agnostic=false) |
| `uint8_t` suppression array | 15–30% speedup vs `vector<bool>` in hot loop |
| OpenMP parallel frames | Near-linear scaling with batch-size (multi-source) |
| Pre-allocated vectors | Eliminates runtime reallocation spikes |

### Example Pipeline (gst-launch)

```bash
gst-launch-1.0 \
  uridecodebin uri=file:///path/to/video.mp4 ! \
  nvstreammux batch-size=1 width=2560 height=1440 ! \
  nvsahipreprocess \
    config-file=preprocess_640.txt \
    slice-width=640 slice-height=640 \
    overlap-width-ratio=0.2 overlap-height-ratio=0.2 \
    enable-full-frame=true ! \
  nvinfer config-file-path=pgie_config.txt input-tensor-meta=1 ! \
  queue ! \
  nvsahipostprocess \
    gie-ids="1" \
    match-metric=1 \
    match-threshold=0.5 \
    class-agnostic=false \
    enable-merge=true \
    two-phase-nmm=true ! \
  nvtracker ... ! \
  nvdsosd ! \
  fakesink
```

### Example (Python — GstElement Properties)

```python
# Pre-process
preprocess = Gst.ElementFactory.make("nvsahipreprocess", "sahi-pre")
preprocess.set_property("config-file", "config/preprocess/preprocess_640.txt")
preprocess.set_property("slice-width", 640)
preprocess.set_property("slice-height", 640)
preprocess.set_property("overlap-width-ratio", 0.2)
preprocess.set_property("overlap-height-ratio", 0.2)
preprocess.set_property("enable-full-frame", True)

# Post-process (v1.2)
postprocess = Gst.ElementFactory.make("nvsahipostprocess", "sahi-post")
postprocess.set_property("gie-ids", "1")
postprocess.set_property("match-metric", 1)             # IoS
postprocess.set_property("match-threshold", 0.5)
postprocess.set_property("class-agnostic", False)
postprocess.set_property("enable-merge", True)
postprocess.set_property("two-phase-nmm", True)          # v1.2
postprocess.set_property("merge-strategy", 0)             # union
postprocess.set_property("max-detections", -1)            # unlimited
postprocess.set_property("drop-mask-on-merge", False)     # composite masks
```

---

## Tuning Guide

### Slice Size

- **Smaller slices** (e.g. 320×320): better for very small objects, but produces more slices per frame → higher GPU cost.
- **Larger slices** (e.g. 960×960): fewer slices, faster, but small objects may still be missed.
- **Rule of thumb**: set `slice-width` and `slice-height` close to the model's native input resolution (e.g. 640 for a 640×640 YOLO model).

### Overlap Ratio

- **0.2** (20%) is a good default. Ensures objects at slice boundaries appear in at least two slices.
- Increase to **0.3–0.4** if you see missed detections at boundaries.
- Higher overlap = more slices = more inference cost.

### Batch Size (network-input-shape[0])

The first dimension of `network-input-shape` in the config file controls how many slices are batched per GPU inference call. Set this to at least the number of slices per frame for optimal throughput. For 9 slices, use `16` (next power-of-two is common for GPU efficiency).

### Match Threshold

- **0.5** is a good starting point for IoS.
- If you see leftover duplicate boxes, lower to **0.3–0.4**.
- If valid nearby detections are being incorrectly merged, raise to **0.6–0.7**.

### Full-Frame Slice

Keep `enable-full-frame=true` (default). Without it, large objects spanning multiple slices may be partially detected in each slice but never seen as a whole. The full-frame slice provides a global view that catches these objects, while GreedyNMM merges the duplicates.

### Two-Phase NMM vs Single-Phase

- **Two-phase** (default, `two-phase-nmm=true`): more conservative. Uses original (immutable) bboxes to select candidates, then re-checks against the expanding bbox. Prevents chain-merging of detections that did not originally overlap.
- **Single-phase** (`two-phase-nmm=false`): more aggressive. The expanding bbox can absorb new detections that the original bbox did not overlap with. Use when you prefer maximally aggressive merging.

### Merge Strategy

- **Union** (default): bbox = min/max of all merged corners. Best general-purpose choice.
- **Weighted**: bbox = confidence-weighted average of coordinates. Produces tighter boxes when both detections are accurate.
- **Largest**: keep the larger bbox unchanged. Useful when you trust the full-frame detection geometry more than slice detections.
