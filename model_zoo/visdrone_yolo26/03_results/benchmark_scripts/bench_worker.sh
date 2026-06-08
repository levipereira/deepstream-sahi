#!/bin/bash
# Runs INSIDE the TensorRT container. Builds an FP16 engine per (model,batch) and benchmarks it.
# Shares a per-model timing cache so only the first batch of a model pays the full tactic search.
# Args: $1 = output CSV (under /work), then a list of jobs "name|onnx|inputname|H|W|batch".
set -u
CSV="$1"; shift
TAG="${GPU_TAG:-?}"
CACHE_DIR=/work/trt_bench/cache;  mkdir -p "$CACHE_DIR"
LOG_DIR=/work/trt_bench/logs;     mkdir -p "$LOG_DIR"

for job in "$@"; do
  IFS='|' read -r name onnx inp H W B <<< "$job"
  shp="${inp}:${B}x3x${H}x${W}"
  cache="$CACHE_DIR/${name}.cache"
  log="$LOG_DIR/${name}_b${B}.log"
  echo "[$TAG] building+benchmarking $name batch=$B ..."
  trtexec --onnx="$onnx" --fp16 \
      --minShapes=$shp --optShapes=$shp --maxShapes=$shp \
      --useCudaGraph --useSpinWait --noDataTransfers \
      --warmUp=2000 --duration=6 --percentile=99 \
      --timingCacheFile="$cache" --memPoolSize=workspace:8192 \
      > "$log" 2>&1
  st=$?
  thr=$(grep -aoP 'Throughput:\s*\K[0-9.]+' "$log" | tail -1)
  gline=$(grep -a 'GPU Compute Time' "$log" | tail -1)
  mean=$(echo "$gline" | grep -oP 'mean = \K[0-9.]+')
  med=$(echo  "$gline" | grep -oP 'median = \K[0-9.]+')
  p99=$(echo  "$gline" | grep -oP 'percentile\([0-9.]+%\) = \K[0-9.]+')
  if [ $st -ne 0 ] || [ -z "$thr" ]; then
      echo "$name,$B,FAIL,,,," >> "$CSV"
      echo "[$TAG] $name b$B -> FAIL (exit $st; see $(basename "$log"))"
  else
      echo "$name,$B,OK,$thr,$mean,$med,$p99" >> "$CSV"
      echo "[$TAG] $name b$B -> thr=${thr} qps  comp_mean=${mean}ms  p99=${p99}ms"
  fi
done
echo "[$TAG] worker done."
