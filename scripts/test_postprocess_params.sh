#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
# nvsahipostprocess — parameter validation test suite
#
# Runs each postprocess parameter variation against a real DeepStream SAHI pipeline
# and extracts per-frame NMM statistics to verify correct behavior.
#
# Usage:
#   ./scripts/test_postprocess_params.sh [VIDEO] [PGIE_CONFIG] [PREPROCESS_CONFIG]
#
# Defaults (from python_test/deepstream-test-sahi):
#   VIDEO             = python_test/videos/aerial_vehicles.mp4
#   PGIE_CONFIG       = python_test/deepstream-test-sahi/config/pgie/visdrone-full-640.txt
#   PREPROCESS_CONFIG = python_test/deepstream-test-sahi/config/preprocess/preprocess_640.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VIDEO="${1:-$ROOT_DIR/python_test/videos/aerial_vehicles.mp4}"
PGIE="${2:-$ROOT_DIR/python_test/deepstream-test-sahi/config/pgie/visdrone-full-640.txt}"
PREPROC="${3:-$ROOT_DIR/python_test/deepstream-test-sahi/config/preprocess/preprocess_640.txt}"

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: Video not found: $VIDEO" >&2; exit 1
fi
if [[ ! -f "$PGIE" ]]; then
  echo "ERROR: PGIE config not found: $PGIE" >&2; exit 1
fi
if [[ ! -f "$PREPROC" ]]; then
  echo "ERROR: Preprocess config not found: $PREPROC" >&2; exit 1
fi

PASS=0
FAIL=0
RESULTS=()

PIPELINE_BASE="filesrc location=$VIDEO ! qtdemux ! h264parse ! nvv4l2decoder \
! m.sink_0 nvstreammux name=m batch-size=1 width=2560 height=1440 \
batched-push-timeout=4000000 \
! nvsahipreprocess config-file=$PREPROC \
  slice-width=640 slice-height=640 \
  overlap-width-ratio=0.2 overlap-height-ratio=0.2 \
  enable-full-frame=true \
! nvinfer config-file-path=$PGIE input-tensor-meta=1 \
! queue"

POSTPROC_SUFFIX="! nvvideoconvert ! nvdsosd ! fakesink"

run_test() {
  local name="$1"
  local props="$2"
  local expect_field="$3"   # "suppressed" | "merged" | "surviving"
  local expect_op="$4"      # "eq" | "gt" | "lt" | "ge" | "le" | "any"
  local expect_val="${5:-0}"

  echo -n "  [$name] ... "

  local cmd="gst-launch-1.0 $PIPELINE_BASE ! nvsahipostprocess $props $POSTPROC_SUFFIX"
  local output
  output=$(GST_DEBUG=nvsahipostprocess:6 eval "$cmd" 2>&1 || true)

  local frame0
  frame0=$(echo "$output" | grep -m1 "process_frame.*frame 0:.*dets," || true)

  if [[ -z "$frame0" ]]; then
    echo "FAIL (no frame 0 output)"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL | $name | no output")
    return
  fi

  local dets suppressed merged surviving
  dets=$(echo "$frame0" | grep -oP '\d+ dets' | grep -oP '^\d+')
  suppressed=$(echo "$frame0" | grep -oP '\d+ suppressed' | grep -oP '^\d+')
  merged=$(echo "$frame0" | grep -oP '\d+ merged' | grep -oP '^\d+')
  surviving=$(echo "$frame0" | grep -oP '\d+ surviving' | grep -oP '^\d+')

  local actual
  case "$expect_field" in
    dets)       actual=$dets ;;
    suppressed) actual=$suppressed ;;
    merged)     actual=$merged ;;
    surviving)  actual=$surviving ;;
  esac

  local ok=false
  case "$expect_op" in
    eq)  [[ "$actual" -eq "$expect_val" ]] && ok=true ;;
    gt)  [[ "$actual" -gt "$expect_val" ]] && ok=true ;;
    lt)  [[ "$actual" -lt "$expect_val" ]] && ok=true ;;
    ge)  [[ "$actual" -ge "$expect_val" ]] && ok=true ;;
    le)  [[ "$actual" -le "$expect_val" ]] && ok=true ;;
    any) ok=true ;;
  esac

  local detail="dets=$dets supp=$suppressed merged=$merged surv=$surviving"
  if $ok; then
    echo "PASS ($detail)"
    PASS=$((PASS + 1))
    RESULTS+=("PASS | $name | $detail")
  else
    echo "FAIL ($detail) — expected $expect_field $expect_op $expect_val, got $actual"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL | $name | $detail | expected $expect_field $expect_op $expect_val")
  fi
}

echo "============================================================"
echo " nvsahipostprocess — Parameter Validation Test Suite"
echo "============================================================"
echo ""
echo "Video:      $VIDEO"
echo "PGIE:       $PGIE"
echo "Preprocess: $PREPROC"
echo ""

# ── Test 1: Baseline defaults ──────────────────────────────────────────────
echo "--- Test 1: Baseline (defaults) ---"
run_test "defaults" \
  'gie-ids="-1" match-metric=1 match-threshold=0.5 class-agnostic=false enable-merge=true two-phase-nmm=true merge-strategy=0 max-detections=-1' \
  surviving gt 0
echo ""

# ── Test 2: match-metric ──────────────────────────────────────────────────
echo "--- Test 2: match-metric (IoU vs IoS) ---"
run_test "metric=0 (IoU)" \
  'gie-ids="-1" match-metric=0 match-threshold=0.5' \
  surviving any
run_test "metric=1 (IoS)" \
  'gie-ids="-1" match-metric=1 match-threshold=0.5' \
  surviving any
echo ""

# ── Test 3: match-threshold ──────────────────────────────────────────────
echo "--- Test 3: match-threshold variations ---"
run_test "threshold=0.3" \
  'gie-ids="-1" match-metric=1 match-threshold=0.3' \
  surviving any
run_test "threshold=0.5" \
  'gie-ids="-1" match-metric=1 match-threshold=0.5' \
  surviving any
run_test "threshold=0.8" \
  'gie-ids="-1" match-metric=1 match-threshold=0.8' \
  surviving any
echo ""

# ── Test 4: class-agnostic ───────────────────────────────────────────────
echo "--- Test 4: class-agnostic ---"
run_test "agnostic=false (per-class)" \
  'gie-ids="-1" class-agnostic=false' \
  surviving any
run_test "agnostic=true (cross-class)" \
  'gie-ids="-1" class-agnostic=true' \
  suppressed gt 146
echo ""

# ── Test 5: enable-merge ─────────────────────────────────────────────────
echo "--- Test 5: enable-merge ---"
run_test "merge=true (NMM)" \
  'gie-ids="-1" enable-merge=true' \
  merged gt 0
run_test "merge=false (NMS only)" \
  'gie-ids="-1" enable-merge=false' \
  merged eq 0
echo ""

# ── Test 6: two-phase-nmm ───────────────────────────────────────────────
echo "--- Test 6: two-phase-nmm ---"
run_test "two-phase=true" \
  'gie-ids="-1" two-phase-nmm=true' \
  surviving any
run_test "two-phase=false" \
  'gie-ids="-1" two-phase-nmm=false' \
  surviving any
echo ""

# ── Test 7: merge-strategy ──────────────────────────────────────────────
echo "--- Test 7: merge-strategy ---"
run_test "strategy=0 (union)" \
  'gie-ids="-1" merge-strategy=0' \
  surviving any
run_test "strategy=1 (weighted)" \
  'gie-ids="-1" merge-strategy=1' \
  surviving any
run_test "strategy=2 (largest)" \
  'gie-ids="-1" merge-strategy=2' \
  surviving any
echo ""

# ── Test 8: max-detections ──────────────────────────────────────────────
echo "--- Test 8: max-detections ---"
run_test "max-det=-1 (unlimited)" \
  'gie-ids="-1" max-detections=-1' \
  surviving gt 100
run_test "max-det=100 (capped)" \
  'gie-ids="-1" max-detections=100' \
  surviving eq 100
echo ""

# ── Test 9: gie-ids ─────────────────────────────────────────────────────
echo "--- Test 9: gie-ids ---"
run_test "gie-ids=-1 (all)" \
  'gie-ids="-1"' \
  surviving any
run_test "gie-ids=1 (specific)" \
  'gie-ids="1"' \
  surviving any
echo ""

# ── Test 10: PERF output ────────────────────────────────────────────────
echo "--- Test 10: PERF latency profiling ---"
echo -n "  [PERF at level 4] ... "
PERF_OUT=$(GST_DEBUG=nvsahipostprocess:4 gst-launch-1.0 $PIPELINE_BASE \
  ! nvsahipostprocess gie-ids="-1" $POSTPROC_SUFFIX 2>&1 || true)
if echo "$PERF_OUT" | grep -q "PERF.*batches.*frames.*avg"; then
  echo "PASS (PERF summary found)"
  PASS=$((PASS + 1))
  RESULTS+=("PASS | PERF at level 4 | summary line present")
else
  echo "FAIL (no PERF output)"
  FAIL=$((FAIL + 1))
  RESULTS+=("FAIL | PERF at level 4 | no PERF output found")
fi
echo ""

# ── Summary ─────────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL))
echo "============================================================"
echo " RESULTS: $PASS/$TOTAL passed, $FAIL failed"
echo "============================================================"
echo ""
for r in "${RESULTS[@]}"; do
  echo "  $r"
done
echo ""

if [[ $FAIL -gt 0 ]]; then
  echo "SOME TESTS FAILED"
  exit 1
else
  echo "ALL TESTS PASSED"
  exit 0
fi
