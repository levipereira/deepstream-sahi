#!/usr/bin/env python3

################################################################################
# SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
# Part of deepstream-sahi — subject to NVIDIA DeepStream SDK License Agreement:
# https://developer.nvidia.com/deepstream-eula
################################################################################
#
# DeepStream Standard Test — Full-Frame Inference (No SAHI Slicing)
#
# Pipeline (default, no tracker):
#   uridecodebin → nvstreammux → nvinfer
#     → nvvidconv → nvdsosd → sink
#
# Pipeline (with --tracker):
#   uridecodebin → nvstreammux → nvinfer → nvtracker
#     → nvvidconv → nvdsosd → sink
#
# Usage:
#   python3 deepstream_test_no_sahi.py --model <model> -i <video_file>
#
# Examples:
#   python3 deepstream_test_no_sahi.py --model visdrone-full-640 -i video.mp4
#   python3 deepstream_test_no_sahi.py --model visdrone-sliced-448 -i video.mp4
#   python3 deepstream_test_no_sahi.py --model visdrone-full-640 --tracker -i video.mp4
#   python3 deepstream_test_no_sahi.py --model visdrone-full-640 --display -i video.mp4
################################################################################

import sys
import argparse

from pipeline_common import (
    MODELS, MUXER_BATCH_TIMEOUT_USEC,
    create_source_bin, path_to_file_uri,
    make_elm_or_die, create_sink, create_mp4_output_bin, configure_tracker,
    osd_sink_pad_buffer_probe,
    add_common_args, validate_resolution_args, resolve_muxer_resolution,
    print_available_models, print_pipeline_chain, CsvLogger,
    PlatformInfo, Gst, GLib, bus_call, PERF_DATA,
)
import pipeline_common


# ─── Pipeline ────────────────────────────────────────────────────────────────

def main(args):
    validate_resolution_args(args)
    muxer_w, muxer_h = resolve_muxer_resolution(args)

    model_cfg = MODELS[args.model]
    pipeline_common.perf_data = PERF_DATA(1)
    pipeline_common.active_model = model_cfg
    platform_info = PlatformInfo()

    print(f"\n=== Standard Pipeline (no SAHI) ===")
    print(f"Model: {args.model} — {model_cfg['description']}")
    print(f"Muxer resolution: {muxer_w}x{muxer_h}")
    print_available_models()

    # ── CSV logger ──
    if args.csv:
        csv_path = (args.csv if args.csv != "auto"
                    else CsvLogger.build_auto_path(args.input, args.model, False))
        pipeline_common.csv_logger = CsvLogger(
            csv_path=csv_path,
            video_path=args.input,
            model_name=args.model,
            sahi_enabled=False,
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
    pgie = make_elm_or_die("nvinfer", "primary-inference")
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

    # ── Configure nvinfer (standard mode — does its own preprocessing) ──
    pgie.set_property("config-file-path", model_cfg["pgie_config"])

    print(f"nvinfer config: {model_cfg['pgie_config']}")
    print(f"Network input: {model_cfg['input_size']}x{model_cfg['input_size']}, "
          f"{model_cfg['num_classes']} classes")

    # ── Configure tracker ──
    if tracker:
        configure_tracker(tracker)
        print("Tracker: enabled")
    else:
        print("Tracker: disabled (default; use --tracker to enable)")

    # ── Assemble pipeline ──
    print("Adding elements to pipeline...")
    elements = [source_bin, streammux, pgie]
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

    streammux.link(pgie)
    if tracker:
        pgie.link(tracker)
        tracker.link(nvvidconv)
    else:
        pgie.link(nvvidconv)
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

    GLib.timeout_add(5000, pipeline_common.perf_data.perf_print_callback)

    # ── Run ──
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("\n*** Starting standard DeepStream pipeline (no SAHI) ***\n")
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
        description="DeepStream Standard Test — full-frame inference (no SAHI slicing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Models:\n" + "\n".join(
            f"  {n:25s} {c['description']}" for n, c in MODELS.items()
        ),
    )
    add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
