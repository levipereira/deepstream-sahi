# TensorRT throughput / latency benchmark (FP16, RTX 4090)

Pure-GPU inference benchmark of YOLO26 ONNX models, swept over batch size to find peak throughput,
the latency/throughput knee, and where latency starts to blow up.

## Setup

| Item | Value |
|------|-------|
| Container | `nvcr.io/nvidia/tensorrt:25.11-py3` (trtexec **v10.14.01**, TensorRT 10.14) |
| GPU | RTX 4090 24 GB |
| Precision | **FP16** (`--fp16`) |
| Measurement flags | `--useCudaGraph --useSpinWait --noDataTransfers --warmUp=2000 --duration=6 --percentile=99` |
| Build | one FP16 engine per (model, batch), static shape = batch (so CUDA graph captures a fixed shape); per-model `--timingCacheFile`, `--memPoolSize=workspace:8192` |
| Batch sweep | 1, 4, 8, 16, 32, 64, 128, 256 |
| Reported latency | **GPU Compute Time** (H2D/D2H = 0 thanks to `--noDataTransfers`), i.e. pure inference |

> `qps` = inferences (batches) per second. **`img/s` = qps × batch** (the throughput that matters).
> Numbers are single-stream pure-GPU compute — real end-to-end adds H2D/D2H + pre/post-processing.

Two measurement passes:
- **Pass 1** — batch sweep (1..256), single stream: `03_results/trt_pass1_batch_sweep.csv` (raw per-run logs in `03_results/raw_trtexec_logs/pass1/`).
- **Pass 2** — multi-stream (streams 1..8) across batch {1, 8, 16, 32}: `03_results/trt_pass2_multistream.csv` (raw logs in `03_results/raw_trtexec_logs/pass2/`).

## Headline summary

| Model | Input | Peak img/s | @batch | Latency @peak | Latency @b1 | Knee batch |
|-------|------:|-----------:|:------:|--------------:|------------:|:----------:|
| **yolo26n** | 416 | **18,356** | 64 | 3.49 ms | 0.43 ms | 16 |
| **yolo26s** | 448 | **7,604**  | 16 | 2.10 ms | 0.54 ms | 16 |

Both models are end-to-end (NMS-free, `[N,6]` output — no separate GPU NMS step).

## Per-model batch sweep

Latency is `GPU Compute Time` mean; `lat/img` = mean / batch (compute efficiency per image).

### yolo26n (input 416)

| batch | img/s | lat mean | lat p99 | lat/img |
|------:|------:|---------:|--------:|--------:|
| 1 | 2,313 | 0.431 ms | 0.431 ms | 0.4310 ms |
| 4 | 7,314 | 0.545 ms | 0.547 ms | 0.1363 ms |
| 8 | 11,389 | 0.701 ms | 0.703 ms | 0.0876 ms |
| **16** | **15,557** | 1.027 ms | 1.029 ms | 0.0642 ms |
| 32 | 18,010 | 1.775 ms | 1.783 ms | 0.0555 ms |
| **64** | **18,356** | 3.485 ms | 3.494 ms | 0.0545 ms |
| 128 | 17,021 | 7.519 ms | 7.541 ms | 0.0587 ms |
| 256 | 15,997 | 16.002 ms | 16.039 ms | 0.0625 ms |

Peak at batch 64 (18,356 img/s). Knee at batch 16: 15,557 img/s at 1.03 ms — 85% of peak at 30% of peak latency.

### yolo26s (input 448)

| batch | img/s | lat mean | lat p99 | lat/img |
|------:|------:|---------:|--------:|--------:|
| 1 | 1,858 | 0.537 ms | 0.538 ms | 0.5370 ms |
| 4 | 4,839 | 0.825 ms | 0.827 ms | 0.2062 ms |
| 8 | 6,576 | 1.215 ms | 1.218 ms | 0.1519 ms |
| **16** | **7,604** | 2.103 ms | 2.108 ms | 0.1314 ms |
| 32 | 7,555 | 4.234 ms | 4.250 ms | 0.1323 ms |
| 64 | 7,057 | 9.067 ms | 9.112 ms | 0.1417 ms |
| 128 | 6,767 | 18.915 ms | 18.993 ms | 0.1478 ms |
| 256 | 6,508 | 39.334 ms | 39.375 ms | 0.1536 ms |

Peak at batch 16 (7,604 img/s). The curve is essentially flat from batch 16 onward — batching beyond 16 adds latency with no throughput benefit.

## Reading the curves

- **Latency is sub-millisecond up to batch 8** (0.43–1.2 ms): at small batch the 4090 is under-utilised (launch/memory bound). The GPU is effectively "free" up to ~batch 8.
- **Throughput peaks at batch 64 for yolo26n and batch 16 for yolo26s.** Beyond the peak, img/s *drops* while latency keeps doubling — pure loss.
- **Per-image latency (`lat/img`) bottoms out at the peak batch**, confirming that point as the compute-efficiency optimum.
- **The latency/throughput knee is batch 16 for both models:** yolo26n delivers 15,557 img/s at 1.03 ms (85% of peak at 30% of peak latency); yolo26s is already at its peak. Batch 16 is the recommended operating point.
- **Latency grows roughly linearly from batch 16 onward** (compute-bound regime).

---

# Pass 2 — multi-stream (`--streams=N`)

Same FP16 / cudaGraph / spinWait / noDataTransfers setup, sweeping **streams ∈ {1, 2, 4, 8}** across
**batch ∈ {1, 8, 16, 32}** (engine reused per model+batch). Goal: find the **max aggregate throughput**
and whether concurrent streams beat single-stream batching.

## Key result: multi-stream does NOT raise peak throughput

| Model | Pass-1 peak (1 stream) | Pass-2 best (any batch×streams) | Best config | Gain |
|-------|----------------------:|--------------------------------:|-------------|:----:|
| yolo26n | 18,356 | 18,542 | batch 32, 2 streams | 1.01× |
| yolo26s | 7,604  | 8,009  | batch 16, 2 streams | 1.05× |

Pass-2 img/s = throughput_qps × batch. yolo26n best: batch 32, 2 streams → 579.43 × 32 = 18,542 img/s.
yolo26s best: batch 16, 2 streams → 500.54 × 16 = 8,009 img/s.

**Conclusion:** at the throughput-optimal batch (16–32) **a single stream already saturates the
RTX 4090's compute**. Adding streams only adds contention — latency rises (often ~2× per stream
doubling) with **no aggregate throughput benefit** (≤5%). For max throughput, **batch, don't stream.**

## Where multi-stream helps: batch = 1 (latency-critical / per-source)

When forced to `batch=1` (e.g. one inference per camera stream, strict per-request latency),
concurrent streams recover most of the lost throughput:

| Model | b1·s1 | b1·s2 | b1·s4 | b1·s8 | lat @s8 | s8/s1 |
|-------|------:|------:|------:|------:|--------:|:-----:|
| yolo26n | 2,340 | 3,823 | 6,444 | 9,014 | 0.86 ms | 3.85× |
| yolo26s | 1,879 | 2,803 | 3,911 | 4,511 | 1.75 ms | 2.40× |

img/s = throughput_qps × 1 (batch=1). Even 8× batch-1 streams (~9,000 img/s for yolo26n) stays
**below** single-stream batch-32 (~18,000 img/s). So: **batch when you can; use streams only when
each request must be batch=1.**

## Practical deployment guidance

- **Offline / high-throughput (can batch):** single stream, **batch 64** (yolo26n) or **batch 16** (yolo26s) = peak img/s; **batch 16** = the latency/throughput knee for both models (~85–100% of peak at roughly half the latency of the peak-batch point). Multi-stream adds nothing here.
- **Online multi-camera (batch=1 per source):** run **N concurrent streams** (≈ number of sources) — ~2.4–3.9× the single-stream batch-1 throughput at sub-2 ms latency, which is the natural DeepStream pattern. yolo26n scales better here (~3.85×) than yolo26s (~2.40×).
- **Recommended operating batch: 16** for both models — delivers ~85–100% of peak throughput at 1–2 ms latency. Use the next-higher batch only if you can absorb the ~2× latency cost.
- Numbers are pure GPU compute. Real-world pipeline throughput also depends on tiling (tiles/frame), detection volume, and the pre/post-processing stages. See `docs/SAHI_MODEL_BENCHMARK.md` for end-to-end pipeline benchmarks.

> Caveats: single-stream, pure GPU compute (no I/O, no pre/post-processing). yolo26 models are end-to-end (no separate NMS step). yolo26s runs at input 448; yolo26n at 416.
