#!/usr/bin/env python3

################################################################################
# SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
# Part of deepstream-sahi — subject to NVIDIA DeepStream SDK License Agreement:
# https://developer.nvidia.com/deepstream-eula
################################################################################
#
# Shared utilities for DeepStream SAHI / no-SAHI test pipelines
#
# Contains:
#   - Model registry (MODELS dict)
#   - Source bin creation (uridecodebin)
#   - GStreamer helpers
#   - Tracker configuration
#   - OSD probe (dynamic class names)
#   - CLI helpers
################################################################################

import sys
import os
import csv
import configparser
from datetime import datetime

_VENV = "/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds"
_ACTIVATE = f"source {_VENV}/bin/activate"

if not os.environ.get("VIRTUAL_ENV"):
    sys.stderr.write(
        f"Error: DeepStream Python virtual environment is not activated.\n"
        f"Run:  {_ACTIVATE}\n"
    )
    sys.exit(1)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    from common.platform_info import PlatformInfo
    from common.bus_call import bus_call
    from common.FPS import PERF_DATA
    import pyds
except ImportError as e:
    sys.stderr.write(
        f"Error: Failed to import required module: {e.name}\n"
        f"Make sure the virtual environment is activated:\n"
        f"  {_ACTIVATE}\n"
    )
    sys.exit(1)


# ─── Model Registry ─────────────────────────────────────────────────────────
#
# Each model defines:
#   pgie_config       — nvinfer config (relative to SCRIPT_DIR)
#   preprocess_config — nvsahipreprocess config (only used in SAHI mode)
#   input_size        — network input W=H in pixels
#   num_classes       — number of detection classes
#   class_names       — list of class label strings
#   default_slice     — recommended slice size for SAHI mode

MODELS = {
    "visdrone-full-640": {
        "description": "VisDrone GELAN-C (full-frame training, 640x640)",
        "pgie_config": "config/pgie/visdrone-full-640.txt",
        "preprocess_config": "config/preprocess/preprocess_640.txt",
        "input_size": 640,
        "num_classes": 11,
        "class_names": [
            "pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor", "others",
        ],
        "class_short": [
            "ped", "ppl", "bik", "car", "van",
            "trk", "tri", "awn", "bus", "mtr", "oth",
        ],
        "default_slice": 640,
    },
    "visdrone-sliced-448": {
        "description": "VisDrone GELAN-C (sliced training, 448x448)",
        "pgie_config": "config/pgie/visdrone-sliced-448.txt",
        "preprocess_config": "config/preprocess/preprocess_448.txt",
        "input_size": 448,
        "num_classes": 11,
        "class_names": [
            "pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor", "others",
        ],
        "class_short": [
            "ped", "ppl", "bik", "car", "van",
            "trk", "tri", "awn", "bus", "mtr", "oth",
        ],
        "default_slice": 448,
    },
}

# ─── Constants ───────────────────────────────────────────────────────────────

RESOLUTION_PRESETS = {
    "4k":    (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p":  (1280, 720),
}

MUXER_BATCH_TIMEOUT_USEC = 33000

TRACKER_CONFIG = "config/tracker/tracker_config.txt"

# ─── OSD Visual Style ────────────────────────────────────────────────────────
# Designed for scenes with many detections (drones, surveillance).
# Thin borders + compact labels = minimal visual pollution.

CLASS_COLORS = [
    (0.0, 1.0, 0.4, 1.0),   # 0  green
    (0.0, 0.8, 1.0, 1.0),   # 1  cyan
    (1.0, 0.85, 0.0, 1.0),  # 2  yellow
    (0.3, 0.6, 1.0, 1.0),   # 3  blue
    (1.0, 0.5, 0.0, 1.0),   # 4  orange
    (1.0, 0.2, 0.3, 1.0),   # 5  red
    (0.8, 0.4, 1.0, 1.0),   # 6  purple
    (1.0, 0.55, 0.75, 1.0), # 7  pink
    (0.4, 1.0, 0.4, 1.0),   # 8  lime
    (0.7, 0.7, 0.7, 1.0),   # 9  gray
    (0.85, 0.75, 0.3, 1.0), # 10 gold
    (0.4, 1.0, 0.85, 1.0),  # 11 teal
]

OSD_BBOX_BORDER_WIDTH = 2
OSD_BBOX_BORDER_ALPHA = 0.85
OSD_LABEL_FONT_SIZE = 8
OSD_LABEL_BG_ALPHA = 0.55
OSD_HUD_FONT_SIZE = 11
OSD_HUD_BG_ALPHA = 0.7

# ─── Globals set by each test script before pipeline starts ──────────────────

perf_data = None
active_model = None  # set to MODELS[name] by the test script
csv_logger = None    # set by CsvLogger.start() if --csv is used


# ─── CSV Logger ──────────────────────────────────────────────────────────────

CSV_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "results")


class CsvLogger:
    """Writes per-frame detection counts to a CSV file for offline analysis.

    Columns:
        video, model, sahi, resolution, tracker, frame, total_objects,
        <class_name_0>, <class_name_1>, …, <class_name_N>
    """

    def __init__(self, csv_path, video_path, model_name, sahi_enabled,
                 resolution_str, tracker_enabled, class_names):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._path = csv_path
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._class_names = list(class_names)
        self._num_classes = len(class_names)

        self._row_prefix = [
            os.path.basename(video_path),
            model_name,
            "sahi" if sahi_enabled else "no_sahi",
            resolution_str,
            "tracker" if tracker_enabled else "no_tracker",
        ]

        header = [
            "video", "model", "sahi", "resolution", "tracker",
            "frame", "total_objects",
        ] + self._class_names
        self._writer.writerow(header)
        self._file.flush()

    def log_frame(self, frame_num, total_objects, obj_counter):
        row = list(self._row_prefix)
        row.append(frame_num)
        row.append(total_objects)
        for i in range(self._num_classes):
            row.append(obj_counter.get(i, 0))
        self._writer.writerow(row)

    def close(self):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            print(f"\nCSV saved: {self._path}")

    @staticmethod
    def build_auto_path(video_path, model_name, sahi_enabled):
        """Generate a descriptive CSV filename under results/."""
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        mode = "sahi" if sahi_enabled else "nosahi"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{video_stem}_{model_name}_{mode}_{ts}.csv"
        return os.path.join(CSV_OUTPUT_DIR, filename)


# ─── Source bin (uridecodebin — auto-detects MP4, MKV, AVI, H264, …) ────────

def _on_pad_added(decodebin, pad, source_bin):
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps(None)
    if not caps or caps.is_empty():
        return
    struct = caps.get_structure(0)
    name = struct.get_name()
    if name and name.startswith("video"):
        ghost_pad = source_bin.get_static_pad("src")
        if not ghost_pad.get_target():
            if ghost_pad.set_target(pad):
                print(f"[uridecodebin] Linked video pad '{pad.get_name()}' to source bin")
            else:
                sys.stderr.write("Failed to set ghost pad target\n")


def create_source_bin(uri):
    source_bin = Gst.Bin.new("source-bin")
    if not source_bin:
        sys.stderr.write("Unable to create source bin\n")
        return None

    uridecodebin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uridecodebin:
        sys.stderr.write("Unable to create uridecodebin\n")
        return None

    uridecodebin.set_property("uri", uri)
    uridecodebin.connect("pad-added", _on_pad_added, source_bin)

    source_bin.add(uridecodebin)
    ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    source_bin.add_pad(ghost_pad)
    return source_bin


def path_to_file_uri(path):
    return "file://" + os.path.abspath(path)


# ─── GStreamer Helpers ───────────────────────────────────────────────────────

def make_elm_or_die(factory, name):
    elm = Gst.ElementFactory.make(factory, name)
    if not elm:
        sys.stderr.write(f"Unable to create {factory} ({name})\n")
        sys.exit(1)
    return elm


def create_sink(platform_info, no_display=False):
    if no_display:
        print("Using fakesink (no display)")
        return make_elm_or_die("fakesink", "fakesink")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        sys.stderr.write(
            "WARNING: --display requested but no DISPLAY or WAYLAND_DISPLAY "
            "environment variable set.\n"
            "  The pipeline will likely hang. On WSL2, use WSLg or X11 forwarding,\n"
            "  or remove --display to use fakesink.\n"
            "  Falling back to fakesink.\n"
        )
        return make_elm_or_die("fakesink", "fakesink")
    if platform_info.is_integrated_gpu() or platform_info.is_platform_aarch64():
        print("Creating nv3dsink")
        return make_elm_or_die("nv3dsink", "nv3d-sink")
    print("Creating EGLSink")
    return make_elm_or_die("nveglglessink", "nvvideo-renderer")


def create_mp4_output_bin(output_path, bitrate=4000000):
    """
    Returns (bin, sink_pad) where the bin encodes NVMM frames to H264
    and writes an MP4 file.  Pipeline inside the bin:
      nvvideoconvert → capsfilter(I420) → nvv4l2h264enc → h264parse → qtmux → filesink
    """
    obin = Gst.Bin.new("mp4-output-bin")

    vidconv = Gst.ElementFactory.make("nvvideoconvert", "mp4-vidconv")
    capsfilter = Gst.ElementFactory.make("capsfilter", "mp4-capsfilter")
    capsfilter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "mp4-encoder")
    encoder.set_property("bitrate", bitrate)
    encoder.set_property("insert-sps-pps", 1)
    parser = Gst.ElementFactory.make("h264parse", "mp4-h264parse")
    muxer = Gst.ElementFactory.make("qtmux", "mp4-qtmux")
    filesink = Gst.ElementFactory.make("filesink", "mp4-filesink")
    filesink.set_property("location", str(output_path))
    filesink.set_property("sync", 0)
    filesink.set_property("async", 0)

    for elm in [vidconv, capsfilter, encoder, parser, muxer, filesink]:
        if not elm:
            sys.stderr.write("Failed to create MP4 output element\n")
            sys.exit(1)
        obin.add(elm)

    vidconv.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)

    ghost_pad = Gst.GhostPad.new("sink", vidconv.get_static_pad("sink"))
    obin.add_pad(ghost_pad)

    print(f"MP4 output: {output_path} (H264 hw-enc, bitrate={bitrate})")
    return obin


# ─── Tracker Configuration ──────────────────────────────────────────────────

def configure_tracker(tracker, config_path=None):
    if config_path is None:
        config_path = os.path.join(_SCRIPT_DIR, TRACKER_CONFIG)
    config = configparser.ConfigParser()
    config.read(config_path)
    for key in config["tracker"]:
        if key == "tracker-width":
            tracker.set_property("tracker-width", config.getint("tracker", key))
        elif key == "tracker-height":
            tracker.set_property("tracker-height", config.getint("tracker", key))
        elif key == "gpu-id":
            tracker.set_property("gpu_id", config.getint("tracker", key))
        elif key == "ll-lib-file":
            tracker.set_property("ll-lib-file", config.get("tracker", key))
        elif key == "ll-config-file":
            tracker.set_property("ll-config-file", config.get("tracker", key))


# ─── OSD Probe (dynamic class names from active_model) ──────────────────────


def _get_class_color(class_id):
    """Return (r, g, b) for a given class_id from the palette."""
    c = CLASS_COLORS[class_id % len(CLASS_COLORS)]
    return c[0], c[1], c[2]


def osd_sink_pad_buffer_probe(pad, info, u_data):
    global active_model, perf_data, csv_logger

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    class_short = active_model.get("class_short", active_model["class_names"])
    num_classes = active_model["num_classes"]

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        obj_counter = {i: 0 for i in range(num_classes)}

        # ── Style each detected object ──
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            cid = obj_meta.class_id
            if cid in obj_counter:
                obj_counter[cid] += 1

            r, g, b = _get_class_color(cid)

            rect = obj_meta.rect_params
            rect.border_width = OSD_BBOX_BORDER_WIDTH
            rect.border_color.set(r, g, b, OSD_BBOX_BORDER_ALPHA)
            rect.has_bg_color = 0

            short = class_short[cid] if cid < len(class_short) else f"c{cid}"
            conf = obj_meta.confidence
            label = f"{short} {int(conf * 100)}" if conf > 0 else short

            txt = obj_meta.text_params
            txt.display_text = label
            txt.x_offset = max(0, int(rect.left))
            txt.y_offset = max(0, int(rect.top) - 12)
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = OSD_LABEL_FONT_SIZE
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(r * 0.4, g * 0.4, b * 0.4, OSD_LABEL_BG_ALPHA)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # ── CSV: log per-frame detection counts ──
        if csv_logger:
            csv_logger.log_frame(frame_number, num_rects, obj_counter)

        # ── HUD: frame summary at top-left ──
        counts_parts = [
            f"{class_short[i]}={obj_counter[i]}"
            for i in range(num_classes) if obj_counter[i] > 0
        ]
        counts_str = " ".join(counts_parts) if counts_parts else "(none)"
        hud_text = f"F={frame_number} Obj={num_rects} {counts_str}"

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        hud = display_meta.text_params[0]
        hud.display_text = hud_text
        hud.x_offset = 10
        hud.y_offset = 12
        hud.font_params.font_name = "Serif"
        hud.font_params.font_size = OSD_HUD_FONT_SIZE
        hud.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        hud.set_bg_clr = 1
        hud.text_bg_clr.set(0.0, 0.0, 0.0, OSD_HUD_BG_ALPHA)
        print(pyds.get_string(hud.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        stream_index = f"stream{frame_meta.pad_index}"
        perf_data.update_fps(stream_index)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ─── Slice count computation (for SAHI validation) ──────────────────────────

def compute_expected_slices(img_w, img_h, slice_w, slice_h,
                            overlap_w, overlap_h, add_full):
    w_overlap = int(slice_w * overlap_w)
    h_overlap = int(slice_h * overlap_h)
    w_step = (slice_w - w_overlap) if slice_w > w_overlap else 1
    h_step = (slice_h - h_overlap) if slice_h > h_overlap else 1
    count = 0
    y = 0
    while y < img_h:
        y_end = min(y + slice_h, img_h)
        y_start = y
        if y_end == img_h and y_start > 0 and (y_end - y_start) < slice_h:
            y_start = (img_h - slice_h) if img_h > slice_h else 0
        x = 0
        while x < img_w:
            x_end = min(x + slice_w, img_w)
            x_start = x
            if x_end == img_w and x_start > 0 and (x_end - x_start) < slice_w:
                x_start = (img_w - slice_w) if img_w > slice_w else 0
            count += 1
            if x_end >= img_w:
                break
            x += w_step
        if y_end >= img_h:
            break
        y += h_step
    if add_full:
        count += 1
    return count


# ─── CLI Helpers ─────────────────────────────────────────────────────────────

def resolve_muxer_resolution(args):
    """Return (width, height) from --resolution preset or --muxer-width/--muxer-height."""
    if args.resolution:
        return RESOLUTION_PRESETS[args.resolution]
    return (args.muxer_width, args.muxer_height)


def add_common_args(parser):
    """Add arguments shared by both SAHI and no-SAHI test scripts."""
    parser.add_argument("-i", "--input", required=True,
                        help="Path to video file (MP4, MKV, AVI, H264, …)")
    parser.add_argument(
        "--model", required=True,
        choices=list(MODELS.keys()),
        help="Model to use for inference",
    )

    res_group = parser.add_mutually_exclusive_group()
    res_group.add_argument(
        "--resolution",
        choices=list(RESOLUTION_PRESETS.keys()),
        default=None,
        help="Muxer output resolution preset: 4k (3840x2160), 1440p (2560x1440), "
             "1080p (1920x1080), 720p (1280x720)",
    )
    res_group.add_argument(
        "--muxer-width", type=int, default=None,
        help="Custom muxer output width (use with --muxer-height)",
    )
    parser.add_argument(
        "--muxer-height", type=int, default=None,
        help="Custom muxer output height (use with --muxer-width)",
    )

    parser.add_argument("--tracker", dest="no_tracker", action="store_false",
                        help="Enable nvtracker (object tracking)")
    parser.add_argument("--no-tracker", dest="no_tracker", action="store_true",
                        help="Disable nvtracker (default)")
    parser.set_defaults(no_tracker=True)

    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", dest="display", action="store_true",
                               help="Enable display sink (EGLSink / nv3dsink)")
    display_group.add_argument("--no-display", dest="display", action="store_false",
                               help="Use fakesink (default)")
    parser.set_defaults(display=False)
    parser.add_argument(
        "--output-mp4", type=str, default=None, metavar="PATH",
        help="Save output video as MP4 (H264 hw-encoded). Implies --no-display.",
    )
    parser.add_argument(
        "--bitrate", type=int, default=15000000,
        help="H264 encoder bitrate in bps (default: 15000000)",
    )
    parser.add_argument(
        "--csv", nargs="?", const="auto", default=None, metavar="PATH",
        help="Export per-frame detections to CSV. "
             "Without value: auto-generate filename in results/. "
             "With path: write to that file.",
    )


def validate_resolution_args(args):
    """Apply defaults and validate --muxer-width / --muxer-height consistency."""
    if args.resolution is None and args.muxer_width is None and args.muxer_height is None:
        args.resolution = "1440p"

    if args.muxer_width is not None or args.muxer_height is not None:
        if args.muxer_width is None or args.muxer_height is None:
            sys.stderr.write("Error: --muxer-width and --muxer-height must be used together\n")
            sys.exit(1)


def print_pipeline_chain(elements):
    """Print the linked pipeline elements as a chain."""
    print(f"\nPipeline chain ({len(elements)} elements):")
    for i, elm in enumerate(elements):
        name = elm.get_name() if hasattr(elm, "get_name") else str(elm)
        factory = elm.get_factory()
        plugin = factory.get_name() if factory else "?"
        label = plugin if plugin == name else f"{plugin}({name})"
        prefix = "  └─" if i == len(elements) - 1 else "  ├─"
        print(f"{prefix} {i+1:2d}. {label}")
    print()


def print_available_models():
    print("\nAvailable models:")
    for name, cfg in MODELS.items():
        print(f"  {name:25s} — {cfg['description']}")
    print()


def print_resolution_presets():
    print("Resolution presets:")
    for name, (w, h) in RESOLUTION_PRESETS.items():
        print(f"  {name:8s} — {w}x{h}")
    print()
