#!/bin/bash
# Pass 2: multi-stream TensorRT throughput sweep across both GPUs in parallel.
# Per (model,batch) engine reused across --streams in {1,2,4,8}. FP16, cudaGraph, spinWait, noDataTransfers.
set -u
cd "$(dirname "$0")/../.."
BUNDLE="$PWD"
IMG=nvcr.io/nvidia/tensorrt:25.11-py3
OUT="$BUNDLE/trt_bench_streams"
mkdir -p "$OUT/cache" "$OUT/engines" "$OUT/logs"

BATCHES="1 8 16 32"

declare -A SPEC=(
  [yolo26n]="04_models/yolo26n/best.onnx|images|416|416"
  [yolo26s]="04_models/yolo26s/best.onnx|images|448|448"
)
GPU0_MODELS="yolo26n"
GPU1_MODELS="yolo26s"

build_jobs () {
  for m in $1; do
    IFS='|' read -r onnx inp H W <<< "${SPEC[$m]}"
    for b in $BATCHES; do echo "${m}|${onnx}|${inp}|${H}|${W}|${b}"; done
  done
}
mapfile -t G0 < <(build_jobs "$GPU0_MODELS")
mapfile -t G1 < <(build_jobs "$GPU1_MODELS")

CSV0="$OUT/results_gpu0.csv"; CSV1="$OUT/results_gpu1.csv"
HDR="model,batch,streams,throughput_qps,gpu_compute_mean_ms,gpu_compute_p99_ms"
echo "$HDR" > "$CSV0"; echo "$HDR" > "$CSV1"

run_gpu () {
  local dev="$1" csv="$2"; shift 2
  docker run --rm --gpus "device=$dev" -e GPU_TAG="GPU$dev" \
    -v "$BUNDLE":/work -w /work "$IMG" \
    bash /work/03_results/benchmark_scripts/bench_worker_streams.sh "$csv" "$@"
}

echo "Pass2 launching GPU0 (${GPU0_MODELS}) and GPU1 (${GPU1_MODELS})..."
run_gpu 0 "/work/trt_bench_streams/results_gpu0.csv" "${G0[@]}" > "$OUT/gpu0.log" 2>&1 &
P0=$!
run_gpu 1 "/work/trt_bench_streams/results_gpu1.csv" "${G1[@]}" > "$OUT/gpu1.log" 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
{ cat "$CSV0"; tail -n +2 "$CSV1"; } > "$OUT/results_all.csv"
echo "GPU0 exit=$R0  GPU1 exit=$R1"
echo "Aggregated -> $OUT/results_all.csv"
