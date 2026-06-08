#!/bin/bash
# Pass 2 (multi-stream). Runs INSIDE the TRT container.
# Builds ONE FP16 engine per (model,batch), then benchmarks it at --streams in {1,2,4,8}
# to find max aggregate throughput. Args: $1 = CSV, then jobs "name|onnx|inp|H|W|batch".
set -u
CSV="$1"; shift
TAG="${GPU_TAG:-?}"
ROOT=/work/trt_bench_streams
mkdir -p "$ROOT/cache" "$ROOT/engines" "$ROOT/logs"
STREAMS="1 2 4 8"

for job in "$@"; do
  IFS='|' read -r name onnx inp H W B <<< "$job"
  shp="${inp}:${B}x3x${H}x${W}"
  eng="$ROOT/engines/${name}_b${B}.engine"
  blog="$ROOT/logs/${name}_b${B}_build.log"
  echo "[$TAG] building engine $name batch=$B ..."
  trtexec --onnx="$onnx" --fp16 \
      --minShapes=$shp --optShapes=$shp --maxShapes=$shp \
      --saveEngine="$eng" --timingCacheFile="$ROOT/cache/${name}.cache" \
      --memPoolSize=workspace:8192 --skipInference > "$blog" 2>&1
  if [ ! -f "$eng" ]; then
      echo "$name,$B,,BUILD_FAIL,," >> "$CSV"; echo "[$TAG] $name b$B BUILD_FAIL"; continue
  fi
  for S in $STREAMS; do
    log="$ROOT/logs/${name}_b${B}_s${S}.log"
    trtexec --loadEngine="$eng" \
        --streams=$S --useCudaGraph --useSpinWait --noDataTransfers \
        --warmUp=2000 --duration=6 --percentile=99 > "$log" 2>&1
    st=$?
    thr=$(grep -aoP 'Throughput:\s*\K[0-9.]+' "$log" | tail -1)
    gline=$(grep -a 'GPU Compute Time:' "$log" | grep -v 'Total' | tail -1)
    mean=$(echo "$gline" | grep -oP 'mean = \K[0-9.]+')
    p99=$(echo  "$gline" | grep -oP 'percentile\([0-9.]+%\) = \K[0-9.]+')
    if [ $st -ne 0 ] || [ -z "$thr" ]; then
        echo "$name,$B,$S,FAIL,," >> "$CSV"; echo "[$TAG] $name b$B s$S -> FAIL"
    else
        echo "$name,$B,$S,$thr,$mean,$p99" >> "$CSV"
        echo "[$TAG] $name b$B s$S -> thr=${thr}qps  comp_mean=${mean}ms  p99=${p99}ms"
    fi
  done
  rm -f "$eng"     # free disk before next engine
done
echo "[$TAG] streams worker done."
