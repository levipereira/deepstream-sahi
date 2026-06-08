#!/bin/bash
# TensorRT throughput/latency benchmark across both GPUs in parallel.
# FP16, --useCudaGraph --useSpinWait --noDataTransfers, batch sweep 1..256.
# One model per GPU (all batches), sharing a per-model timing cache.
set -u
cd "$(dirname "$0")/../.."          # -> bundle root (visdrone_benchmark/)
BUNDLE="$PWD"
IMG=nvcr.io/nvidia/tensorrt:25.11-py3
OUT="$BUNDLE/trt_bench"
mkdir -p "$OUT/cache" "$OUT/logs"

BATCHES="1 4 8 16 32 64 128 256"

# model | onnx (relative to bundle) | input-tensor | H | W
declare -A SPEC=(
  [yolo26n]="04_models/yolo26n/best.onnx|images|416|416"
  [yolo26s]="04_models/yolo26s/best.onnx|images|448|448"
)

# One model per GPU.
GPU0_MODELS="yolo26n"
GPU1_MODELS="yolo26s"

build_jobs () {           # $1 = space-separated model names -> prints job strings
  for m in $1; do
    IFS='|' read -r onnx inp H W <<< "${SPEC[$m]}"
    for b in $BATCHES; do echo "${m}|${onnx}|${inp}|${H}|${W}|${b}"; done
  done
}
mapfile -t G0 < <(build_jobs "$GPU0_MODELS")
mapfile -t G1 < <(build_jobs "$GPU1_MODELS")

CSV0="$OUT/results_gpu0.csv"; CSV1="$OUT/results_gpu1.csv"
HDR="model,batch,status,throughput_qps,gpu_compute_mean_ms,gpu_compute_median_ms,gpu_compute_p99_ms"
echo "$HDR" > "$CSV0"; echo "$HDR" > "$CSV1"

run_gpu () {              # $1=device  $2=csv(container path)  rest=jobs
  local dev="$1" csv="$2"; shift 2
  docker run --rm --gpus "device=$dev" -e GPU_TAG="GPU$dev" \
    -v "$BUNDLE":/work -w /work "$IMG" \
    bash /work/03_results/benchmark_scripts/bench_worker.sh "$csv" "$@"
}

echo "Launching GPU0 (${GPU0_MODELS}) and GPU1 (${GPU1_MODELS}) in parallel..."
run_gpu 0 "/work/trt_bench/results_gpu0.csv" "${G0[@]}" > "$OUT/gpu0.log" 2>&1 &
P0=$!
run_gpu 1 "/work/trt_bench/results_gpu1.csv" "${G1[@]}" > "$OUT/gpu1.log" 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?

# aggregate
{ cat "$CSV0"; tail -n +2 "$CSV1"; } > "$OUT/results_all.csv"
echo "GPU0 exit=$R0  GPU1 exit=$R1"
echo "Aggregated -> $OUT/results_all.csv"
