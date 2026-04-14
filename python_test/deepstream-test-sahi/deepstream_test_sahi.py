#!/usr/bin/env python3

################################################################################
# SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
# Part of deepstream-sahi — subject to NVIDIA DeepStream SDK License Agreement:
# https://developer.nvidia.com/deepstream-eula
################################################################################
#
# DeepStream SAHI Test — Sliced Inference with GreedyNMM Merging
#
# Pipeline (default, no tracker):
#   uridecodebin → nvstreammux
#     → nvsahipreprocess → nvinfer (input-tensor-meta)
#     → queue → nvsahipostprocess
#     → nvvidconv → nvdsosd → sink
#
# Pipeline (with --tracker):
#   uridecodebin → nvstreammux
#     → nvsahipreprocess → nvinfer (input-tensor-meta)
#     → queue → nvsahipostprocess
#     → nvtracker → nvvidconv → nvdsosd → sink
#
# Usage:
#   python3 deepstream_test_sahi.py --model <model> -i <video_file>
#
# Examples:
#   python3 deepstream_test_sahi.py --model visdrone-full-640 -i video.mp4
#   python3 deepstream_test_sahi.py --model visdrone-sliced-448 -i video.mp4
#   python3 deepstream_test_sahi.py --model visdrone-sliced-448 --tracker -i video.mp4
#   python3 deepstream_test_sahi.py --model visdrone-sliced-448 \
#       --slice-width 448 --slice-height 448 --overlap-w 0.2 -i video.mp4
#
# Optional SAHI arguments:
#   --slice-width       Slice width in pixels  (default: model's input_size)
#   --slice-height      Slice height in pixels (default: model's input_size)
#   --overlap-w         Horizontal overlap ratio 0.0–0.99 (default: 0.2)
#   --overlap-h         Vertical overlap ratio 0.0–0.99   (default: 0.2)
#   --no-full-frame     Disable full-frame as extra slice
#   --match-metric      NMM metric: 0=IoU, 1=IoS (default: 1)
#   --match-threshold   NMM overlap threshold (default: 0.5)
#   --display           Enable display sink (EGLSink / nv3dsink)
#   --no-display        Run without display (fakesink, default)
#   --validate-slices   Print per-frame slice validation
################################################################################

import sys
import argparse

from pipeline_common import (
    MODELS, MUXER_BATCH_TIMEOUT_USEC,
    create_source_bin, path_to_file_uri,
    make_elm_or_die, create_sink, create_mp4_output_bin, configure_tracker,
    osd_sink_pad_buffer_probe, compute_expected_slices,
    add_common_args, validate_resolution_args, resolve_muxer_resolution,
    print_available_models, print_pipeline_chain, CsvLogger,
    PlatformInfo, Gst, GLib, bus_call, PERF_DATA, pyds,
)
import pipeline_common


# ─── Slice validation state ─────────────────────────────────────────────────

_slice_validation = {"roi_count": {}, "expected": None}


def pgie_sink_pad_slice_validation_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    if l_frame is None:
        return Gst.PadProbeReturn.OK

    try:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
    except StopIteration:
        return Gst.PadProbeReturn.OK

    frame_num = frame_meta.frame_num
    source_id = frame_meta.pad_index

    roi_per_frame_this_buf = {}
    l_user = batch_meta.batch_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if user_meta.base_meta.meta_type == pyds.NVDS_PREPROCESS_BATCH_META:
            try:
                pp_batch = pyds.GstNvDsPreProcessBatchMeta.cast(user_meta.user_meta_data)
                for roi_meta in pp_batch.roi_vector:
                    try:
                        fm = roi_meta.frame_meta
                        k = (fm.source_id, fm.frame_num)
                        roi_per_frame_this_buf[k] = roi_per_frame_this_buf.get(k, 0) + 1
                    except (StopIteration, AttributeError, TypeError):
                        pass
            except (StopIteration, AttributeError, TypeError):
                pass
        try:
            l_user = l_user.next
        except StopIteration:
            break

    if not roi_per_frame_this_buf:
        return Gst.PadProbeReturn.OK

    sv = _slice_validation
    for k, cnt in roi_per_frame_this_buf.items():
        sv["roi_count"][k] = sv["roi_count"].get(k, 0) + cnt

    prev_key = (source_id, sv.get("last_frame_num", -1))
    if prev_key[1] >= 0 and frame_num != prev_key[1]:
        total = sv["roi_count"].get(prev_key, 0)
        expected = sv["expected"]
        ok = "OK" if total == expected else f"MISMATCH (expected {expected})"
        print(f"[SLICE_VALID] frame={prev_key[1]} total_rois={total} {ok}")
        if prev_key in sv["roi_count"]:
            del sv["roi_count"][prev_key]

    if frame_num > sv.get("last_frame_num", -1):
        sv["last_frame_num"] = frame_num
    return Gst.PadProbeReturn.OK


# ─── Pipeline ────────────────────────────────────────────────────────────────

def main(args):
    validate_resolution_args(args)
    muxer_w, muxer_h = resolve_muxer_resolution(args)

    model_cfg = MODELS[args.model]
    pipeline_common.perf_data = PERF_DATA(1)
    pipeline_common.active_model = model_cfg
    platform_info = PlatformInfo()

    print(f"\n=== SAHI Pipeline ===")
    print(f"Model: {args.model} — {model_cfg['description']}")
    print(f"Muxer resolution: {muxer_w}x{muxer_h}")
    print_available_models()

    # ── CSV logger ──
    if args.csv:
        csv_path = (args.csv if args.csv != "auto"
                    else CsvLogger.build_auto_path(args.input, args.model, True))
        pipeline_common.csv_logger = CsvLogger(
            csv_path=csv_path,
            video_path=args.input,
            model_name=args.model,
            sahi_enabled=True,
            resolution_str=f"{muxer_w}x{muxer_h}",
            tracker_enabled=not args.no_tracker,
            class_names=model_cfg["class_names"],
        )
        print(f"CSV output: {csv_path}")

    Gst.init(None)

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        sys.exit(1)

    # ── Source ──
    uri = path_to_file_uri(args.input)
    source_bin = create_source_bin(uri)
    if not source_bin:
        sys.exit(1)

    streammux = make_elm_or_die("nvstreammux", "stream-muxer")
    sahipreprocess = make_elm_or_die("nvsahipreprocess", "sahi-preprocess")
    pgie = make_elm_or_die("nvinfer", "primary-inference")
    queue_post = make_elm_or_die("queue", "queue-postprocess")
    sahipostprocess = make_elm_or_die("nvsahipostprocess", "sahi-postprocess")
    tracker = None if args.no_tracker else make_elm_or_die("nvtracker", "tracker")
    nvvidconv = make_elm_or_die("nvvideoconvert", "convertor")
    nvosd = make_elm_or_die("nvdsosd", "onscreendisplay")

    if args.output_mp4:
        mp4bin = create_mp4_output_bin(args.output_mp4, args.bitrate)
        sink = create_sink(platform_info, no_display=True)
    else:
        mp4bin = None
        sink = create_sink(platform_info, no_display=not args.display)

    print(f"Playing file: {args.input} (uri={uri})")

    # ── Configure streammux ──
    streammux.set_property("width", muxer_w)
    streammux.set_property("height", muxer_h)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

    # ── Configure nvsahipreprocess ──
    slice_w = args.slice_width if args.slice_width else model_cfg["default_slice"]
    slice_h = args.slice_height if args.slice_height else model_cfg["default_slice"]

    sahipreprocess.set_property("config-file", model_cfg["preprocess_config"])
    sahipreprocess.set_property("slice-width", slice_w)
    sahipreprocess.set_property("slice-height", slice_h)
    sahipreprocess.set_property("overlap-width-ratio", args.overlap_w)
    sahipreprocess.set_property("overlap-height-ratio", args.overlap_h)
    full_frame = not args.no_full_frame
    sahipreprocess.set_property("enable-full-frame", full_frame)

    print(f"SAHI slicing: {slice_w}x{slice_h} "
          f"overlap=({args.overlap_w:.2f}, {args.overlap_h:.2f}) "
          f"full-frame={full_frame}")

    # ── Slice validation ──
    if args.validate_slices:
        expected = compute_expected_slices(
            muxer_w, muxer_h,
            slice_w, slice_h,
            args.overlap_w, args.overlap_h, full_frame,
        )
        _slice_validation["expected"] = expected
        _slice_validation["roi_count"] = {}
        _slice_validation["last_frame_num"] = -1
        print(f"[SLICE_VALID] Expected slices per frame: {expected}")

    # ── Configure nvinfer (receives tensors from nvsahipreprocess) ──
    pgie.set_property("config-file-path", model_cfg["pgie_config"])
    pgie.set_property("input-tensor-meta", True)

    # ── Configure nvsahipostprocess (GreedyNMM) ──
    sahipostprocess.set_property("match-metric", args.match_metric)
    sahipostprocess.set_property("match-threshold", args.match_threshold)
    sahipostprocess.set_property("class-agnostic", False)
    sahipostprocess.set_property("enable-merge", True)

    metric_name = "IoS" if args.match_metric == 1 else "IoU"
    print(f"SAHI postprocess: GreedyNMM metric={metric_name} "
          f"threshold={args.match_threshold:.2f}")

    # ── Configure tracker ──
    if tracker:
        configure_tracker(tracker)
        print("Tracker: enabled")
    else:
        print("Tracker: disabled (default; use --tracker to enable)")

    # ── Assemble pipeline ──
    print("Adding elements to pipeline...")
    elements = [source_bin, streammux,
                sahipreprocess, pgie, queue_post, sahipostprocess]
    if tracker:
        elements.append(tracker)
    elements.extend([nvvidconv, nvosd])
    if mp4bin:
        tee = make_elm_or_die("tee", "osd-tee")
        queue_display = make_elm_or_die("queue", "queue-display")
        queue_mp4 = make_elm_or_die("queue", "queue-mp4")
        elements.extend([tee, queue_display, sink, queue_mp4, mp4bin])
    else:
        elements.append(sink)
    for elm in elements:
        pipeline.add(elm)

    sinkpad = streammux.request_pad_simple("sink_0")
    srcpad = source_bin.get_static_pad("src")
    srcpad.link(sinkpad)

    streammux.link(sahipreprocess)
    sahipreprocess.link(pgie)
    pgie.link(queue_post)
    queue_post.link(sahipostprocess)
    if tracker:
        sahipostprocess.link(tracker)
        tracker.link(nvvidconv)
    else:
        sahipostprocess.link(nvvidconv)
    nvvidconv.link(nvosd)

    if mp4bin:
        nvosd.link(tee)
        tee.link(queue_display)
        queue_display.link(sink)
        tee.link(queue_mp4)
        queue_mp4.link(mp4bin)
    else:
        nvosd.link(sink)

    print("Pipeline linked successfully")
    print_pipeline_chain(elements)

    # ── Probes ──
    osdsinkpad = nvosd.get_static_pad("sink")
    if osdsinkpad:
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    if args.validate_slices:
        pgie_sink = pgie.get_static_pad("sink")
        if pgie_sink:
            pgie_sink.add_probe(
                Gst.PadProbeType.BUFFER,
                pgie_sink_pad_slice_validation_probe, 0,
            )

    GLib.timeout_add(5000, pipeline_common.perf_data.perf_print_callback)

    # ── Run ──
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("\n*** Starting SAHI DeepStream pipeline ***\n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except BaseException:
        pass

    pipeline.set_state(Gst.State.NULL)
    if pipeline_common.csv_logger:
        pipeline_common.csv_logger.close()
    print("Pipeline stopped.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepStream SAHI Test — sliced inference with GreedyNMM merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Models:\n" + "\n".join(
            f"  {n:25s} {c['description']}" for n, c in MODELS.items()
        ),
    )
    add_common_args(parser)

    parser.add_argument("--slice-width", type=int, default=None,
                        help="SAHI slice width in pixels (default: model's input size)")
    parser.add_argument("--slice-height", type=int, default=None,
                        help="SAHI slice height in pixels (default: model's input size)")
    parser.add_argument("--overlap-w", type=float, default=0.2,
                        help="Horizontal overlap ratio (default: 0.2)")
    parser.add_argument("--overlap-h", type=float, default=0.2,
                        help="Vertical overlap ratio (default: 0.2)")
    parser.add_argument("--no-full-frame", action="store_true",
                        help="Disable full-frame slice")
    parser.add_argument("--match-metric", type=int, default=1, choices=[0, 1],
                        help="NMM metric: 0=IoU, 1=IoS (default: 1)")
    parser.add_argument("--match-threshold", type=float, default=0.5,
                        help="NMM overlap threshold (default: 0.5)")
    parser.add_argument("--validate-slices", action="store_true",
                        help="Validate that all SAHI slices undergo inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
