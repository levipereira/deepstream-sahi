# Review — `nvsahipreprocess` & `nvsahipostprocess` (with improvement suggestions)

Careful read of the two SAHI plugins. Prioritised by impact.

> **Measured first (important).** At the shipped operating point (threshold 0.40, ~211
> objects/frame, dense aerial scene), `nvsahipostprocess` costs only **~0.18 ms/frame** (≈3.5%
> of the 5.1 ms budget at ~196 FPS). **It is NOT the current bottleneck.** Tiling (tiles/frame)
> and detection volume dominate — confirmed by the SAHI-on-black test. So the postprocess
> optimizations below are **good hygiene that only pays off at extreme detection counts**
> (e.g. threshold 0.25 → ~1066 obj/frame, where the grid query's `sort+unique` and the
> `cell_size` cliff would dominate). At the operating threshold they are low priority.

## nvsahipreprocess

**Design (good):** per-frame `compute_sahi_slices()` (ported from SAHI `get_slice_bboxes`), then **one
batched** `NvBufSurfTransformAsync` crops+scales all tiles into network-res surfaces (GPU, async), and
a custom acquirer builds the `GstNvDsPreProcessBatchMeta` tensor for `nvinfer` (`input-tensor-meta=1`).
Tiles can exceed the network input (resized down). This is efficient — the tiling itself is not the
bottleneck.

**Suggestions:**
1. **Tiles vs batch awareness (med).** If `tiles/frame > network-input-shape[0]`, nvinfer runs
   `ceil(tiles/batch)` passes (the #1 perf factor). The plugin should **log tiles-per-frame vs the
   configured batch** once, and warn if they differ, so users set `batch = tiles`. (We documented the
   rule; surfacing it here prevents silent 3×-slower configs.)
2. **Cache slices (low).** `compute_sahi_slices` runs every frame though slice params + resolution are
   constant; compute once and reuse (cheap, but free).
3. **PERF counter (low).** Add the same 1 s timing log the postprocess has, to report tiling ms/frame
   (currently not measurable — the GST latency tracer doesn't cover these elements).

## nvsahipostprocess (the hot path)

**Design:** collect `NvDsObjectMeta` → sort by score → build a `SahiSpatialGrid` → two-phase GreedyNMM
(per-class or class-agnostic) → max-det cap → drop suppressed / rewrite merged meta. OpenMP across
**frames** (helps multi-source only; single source = 1 thread).

### ⭐ High-impact

1. **`SahiSpatialGrid::query` sorted+deduped on every call — DONE (implemented).** `query()` gathered
   candidates from overlapping cells then did `std::sort` + `std::unique` — **O(k log k) per detection,
   O(n·k log k) total**. Replaced with a **visited-stamp** (`std::vector<guint> stamp_` + a per-build
   generation counter `gen_`): mark `stamp_[idx]=gen_` when first seen, skip if already `==gen_`. O(k),
   no sort, no allocation. **Measured (thr 0.20, ~1419 obj/frame): postprocess 3.47 → 2.66
   ms/frame, ~23% faster, results identical** (209.7 obj/frame at thr 0.40, unchanged). The grid stays
   exact (each box still emitted once per query), so detections are bit-for-bit the same.

2. **`cell_size_ = max box dimension` — ATTEMPTED, REVERTED.** Tried a median-based, clamped cell size
   (to avoid a theoretical grid collapse when one box is huge). It was faster (2.20 ms) **but changed
   the results** (235.9 vs 209.7 obj/frame — fewer suppressions), so it was **dropped**. In practice
   aerial scenes have small objects (`max_dim` does not blow up), so the grid never actually
   collapses and the original `max_dim` is fine. A correct robust-cell-size variant would need more
   care (the regression was traced to the cell-size change, not the visited-stamp); left for later and
   low priority — postprocess is ~0.18 ms at the operating threshold (not the bottleneck).

### Medium

3. **Per-frame heap churn.** `rects` vector, the grid's `std::vector<std::vector<guint>> cells_`,
   `by_class` `unordered_map`, and `merge_list` `unordered_map` are all allocated every frame.
   - Make the scratch buffers **thread-local and reused** across frames (`build()` already `clear()`s).
   - Replace `cells_` (vector-of-vectors) with a **CSR layout** (flat `counts`/`offsets`/`indices`
     arrays) — one allocation, better cache locality, no per-cell `push_back` growth.
   - Replace `by_class` `unordered_map<gint,vector>` with a **flat `std::array<vector, num_classes>`**
     (num_classes is small, 11) — drop the hash map.
4. **`merge_list` unordered_map** per group → a flat vector keyed by index (or store merges inline on
   the detection) avoids the hash map.

### Low
5. Per-detection `strncpy`/`memcpy` of labels for every box — skip for detection-only models if the
   label isn't consumed downstream.
6. **Optional GPU NMM** for extreme density — only worth it if (1)/(2) don't suffice; the grid is fine
   once the cell-size cliff is removed.

## Why this matters for dense aerial scenes

EfficientNMS-based models move **per-tile** NMS to the GPU, but this gives only modest gains (~4%)
because the dominant cost is the **cross-tile GreedyNMM here**, which scales with the final detection
count. Fixing (1) and (2) attacks that directly and benefits **every** model on dense scenes
(GELAN, YOLO26, and any other detector). Expected: the merge stops being super-linear in dense
frames, so high detection-count operating points (low threshold / many tiles) no longer cause
latency spikes.
