/* SPDX-FileCopyrightText: Copyright (c) 2026 Levi Pereira <levi.pereira@gmail.com>
 * SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
 *
 * Subject to https://developer.nvidia.com/deepstream-eula
 *
 * DeepStream SAHI Post-Process Plugin (v1.2)
 * Merges duplicate detections from sliced inference via GreedyNMM.
 * Pipeline: nvinfer → queue → nvsahipostprocess → nvtracker → nvdsosd
 */

#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <vector>
#include <unordered_map>
#include "greedy_nmm.h"

#ifdef _OPENMP
#include <omp.h>
#endif

GST_DEBUG_CATEGORY_STATIC (gst_nvsahipostprocess_debug);
#define GST_CAT_DEFAULT gst_nvsahipostprocess_debug

enum
{
  PROP_0,
  PROP_GIE_IDS,
  PROP_MATCH_METRIC,
  PROP_MATCH_THRESHOLD,
  PROP_CLASS_AGNOSTIC,
  PROP_ENABLE_MERGE,
  PROP_TWO_PHASE_NMM,
  PROP_MERGE_STRATEGY,
  PROP_MAX_DETECTIONS,
  PROP_DROP_MASK_ON_MERGE,
};

#define DEFAULT_GIE_IDS             "-1"
#define DEFAULT_MATCH_METRIC        SAHI_METRIC_IOS
#define DEFAULT_MATCH_THRESHOLD     0.5f
#define DEFAULT_CLASS_AGNOSTIC      FALSE
#define DEFAULT_ENABLE_MERGE        TRUE
#define DEFAULT_TWO_PHASE_NMM       TRUE
#define DEFAULT_MERGE_STRATEGY      SAHI_MERGE_UNION
#define DEFAULT_MAX_DETECTIONS      -1
#define DEFAULT_DROP_MASK_ON_MERGE  FALSE

#define NVMM_CAPS GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM", "{ NV12, RGBA, I420 }")
static GstStaticPadTemplate sink_tmpl = GST_STATIC_PAD_TEMPLATE (
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS (NVMM_CAPS));
static GstStaticPadTemplate src_tmpl = GST_STATIC_PAD_TEMPLATE (
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS (NVMM_CAPS));

#define gst_nvsahipostprocess_parent_class parent_class
G_DEFINE_TYPE (GstNvSahiPostProcess, gst_nvsahipostprocess, GST_TYPE_BASE_TRANSFORM);

static void gst_nvsahipostprocess_set_property (GObject *, guint, const GValue *, GParamSpec *);
static void gst_nvsahipostprocess_get_property (GObject *, guint, GValue *, GParamSpec *);
static void gst_nvsahipostprocess_finalize (GObject *);
static GstFlowReturn gst_nvsahipostprocess_transform_ip (GstBaseTransform *, GstBuffer *);

static void
parse_gie_ids (GstNvSahiPostProcess *self, const gchar *str)
{
  self->gie_ids->clear ();
  self->gie_filter_all = FALSE;
  g_free (self->gie_ids_str);
  self->gie_ids_str = g_strdup (str ? str : DEFAULT_GIE_IDS);
  gchar **tokens = g_strsplit (self->gie_ids_str, ";", -1);
  for (guint i = 0; tokens[i]; i++) {
    g_strstrip (tokens[i]);
    if (tokens[i][0] == '\0') continue;
    gint val = atoi (tokens[i]);
    if (val < 0) {
      self->gie_filter_all = TRUE;
      self->gie_ids->clear ();
      break;
    }
    self->gie_ids->insert (val);
  }
  g_strfreev (tokens);
  if (self->gie_ids->empty ())
    self->gie_filter_all = TRUE;
}

/* ── Per-frame processing (thread-safe: all state is local) ──────────────── */

static void
process_frame (GstNvSahiPostProcess *self,
    NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta)
{
  std::vector<SahiDetection> dets;
  std::vector<uint8_t> suppressed;
  std::vector<guint> order;
  SahiSpatialGrid grid;

  dets.reserve (512);

  for (NvDsMetaList *l = frame_meta->obj_meta_list; l; l = l->next) {
    NvDsObjectMeta *obj = (NvDsObjectMeta *) l->data;

    if (!self->gie_filter_all &&
        self->gie_ids->find (obj->unique_component_id) == self->gie_ids->end ())
      continue;

    SahiDetection d;
    d.left   = obj->rect_params.left;
    d.top    = obj->rect_params.top;
    d.right  = d.left + obj->rect_params.width;
    d.bottom = d.top  + obj->rect_params.height;
    d.orig_left = d.left; d.orig_top = d.top;
    d.orig_right = d.right; d.orig_bottom = d.bottom;
    d.score    = obj->confidence;
    d.class_id = obj->class_id;
    strncpy (d.obj_label, obj->obj_label, sizeof (d.obj_label) - 1);
    d.obj_label[sizeof (d.obj_label) - 1] = '\0';
    d.area     = obj->rect_params.width * obj->rect_params.height;
    d.obj_meta = obj;
    d.merged   = FALSE;
    d.merged_mask_data = nullptr;
    d.mask = sahi_mask_from_nvds (
        obj->mask_params.data, obj->mask_params.width,
        obj->mask_params.height, obj->mask_params.threshold);
    d.best_score = d.score;
    d.best_class_id = d.class_id;
    memcpy (d.best_label, d.obj_label, sizeof (d.best_label));

    dets.push_back (d);
  }

  GST_LOG_OBJECT (self, "frame %u: collected %zu dets (gie_filter_all=%d)",
      frame_meta->frame_num, dets.size (), self->gie_filter_all);

  if (dets.size () <= 1)
    return;

  const guint n = dets.size ();
  order.resize (n);
  std::iota (order.begin (), order.end (), 0);
  std::sort (order.begin (), order.end (),
      [&dets](guint a, guint b) { return det_compare (dets, a, b); });

  suppressed.assign (n, 0);

  const gfloat threshold = self->match_threshold;
  const gboolean agnostic = self->class_agnostic;
  const gboolean do_merge = self->enable_merge;
  const gboolean two_phase = self->two_phase_nmm;
  const guint metric = self->match_metric;
  const guint strategy = self->merge_strategy;
  const gboolean drop_mask = self->drop_mask_on_merge;

  /* Build spatial index */
  gfloat fw = 0, fh = 0;
  if (frame_meta->source_frame_width > 0 &&
      frame_meta->source_frame_height > 0) {
    fw = (gfloat) frame_meta->source_frame_width;
    fh = (gfloat) frame_meta->source_frame_height;
  } else {
    for (guint i = 0; i < n; i++) {
      fw = MAX (fw, dets[i].right);
      fh = MAX (fh, dets[i].bottom);
    }
  }

  std::vector<SahiGridRect> rects (n);
  for (guint i = 0; i < n; i++)
    rects[i] = {dets[i].left, dets[i].top, dets[i].right, dets[i].bottom};
  grid.build (rects, n, fw, fh);
  GST_LOG_OBJECT (self, "frame %u: grid built %.0fx%.0f, %u rects",
      frame_meta->frame_num, fw, fh, n);

  /* Execute NMM (per-class or class-agnostic) */
  if (!agnostic) {
    std::unordered_map<gint, std::vector<guint>> by_class;
    for (guint ii = 0; ii < order.size (); ii++)
      by_class[dets[order[ii]].class_id].push_back (order[ii]);
    for (auto &kv : by_class)
      run_nmm_on_group (dets, suppressed, grid, kv.second,
          threshold, metric, strategy, agnostic, do_merge, two_phase,
          drop_mask);
  } else {
    std::vector<guint> all (order.begin (), order.end ());
    run_nmm_on_group (dets, suppressed, grid, all,
        threshold, metric, strategy, agnostic, do_merge, two_phase,
        drop_mask);
  }

  /* Max detections cap */
  if (self->max_detections > 0) {
    std::vector<guint> survivors;
    for (guint i = 0; i < n; i++)
      if (!suppressed[i]) survivors.push_back (i);
    if ((gint) survivors.size () > self->max_detections) {
      std::sort (survivors.begin (), survivors.end (),
          [&dets](guint a, guint b) { return dets[a].score > dets[b].score; });
      for (gint k = self->max_detections; k < (gint) survivors.size (); k++)
        suppressed[survivors[k]] = 1;
    }
  }

  /* Debug statistics */
  guint n_suppressed = 0, n_merged = 0;
  for (guint i = 0; i < n; i++) {
    if (suppressed[i]) n_suppressed++;
    else if (dets[i].merged) n_merged++;
  }
  GST_LOG_OBJECT (self,
      "frame %u: %u dets, %u suppressed, %u merged, %u surviving",
      frame_meta->frame_num, n, n_suppressed, n_merged, n - n_suppressed);

  /* Apply to metadata */
  std::vector<NvDsObjectMeta *> to_remove;
  to_remove.reserve (n_suppressed);

  for (guint i = 0; i < n; i++) {
    if (suppressed[i]) {
      to_remove.push_back (dets[i].obj_meta);
    } else if (dets[i].merged) {
      NvDsObjectMeta *obj = dets[i].obj_meta;
      obj->rect_params.left   = dets[i].left;
      obj->rect_params.top    = dets[i].top;
      obj->rect_params.width  = dets[i].right - dets[i].left;
      obj->rect_params.height = dets[i].bottom - dets[i].top;

      obj->detector_bbox_info.org_bbox_coords.left   = dets[i].left;
      obj->detector_bbox_info.org_bbox_coords.top    = dets[i].top;
      obj->detector_bbox_info.org_bbox_coords.width  = obj->rect_params.width;
      obj->detector_bbox_info.org_bbox_coords.height = obj->rect_params.height;

      obj->confidence = dets[i].score;

      if (agnostic && dets[i].best_class_id != obj->class_id) {
        obj->class_id = dets[i].best_class_id;
        strncpy (obj->obj_label, dets[i].best_label,
                 sizeof (obj->obj_label) - 1);
        obj->obj_label[sizeof (obj->obj_label) - 1] = '\0';
      }

      if (dets[i].merged_mask_data && !drop_mask) {
        if (obj->mask_params.data)
          g_free (obj->mask_params.data);
        guint sz = dets[i].mask.width * dets[i].mask.height;
        obj->mask_params.data = (float *) g_malloc (sz * sizeof (float));
        memcpy (obj->mask_params.data, dets[i].merged_mask_data,
                sz * sizeof (float));
        obj->mask_params.width = dets[i].mask.width;
        obj->mask_params.height = dets[i].mask.height;
        obj->mask_params.size = sz * sizeof (float);
        obj->mask_params.threshold = dets[i].mask.threshold;
      } else if (drop_mask && obj->mask_params.data) {
        g_free (obj->mask_params.data);
        obj->mask_params.data = nullptr;
        obj->mask_params.size = 0;
        obj->mask_params.width = 0;
        obj->mask_params.height = 0;
      }
    }
  }

  nvds_acquire_meta_lock (batch_meta);
  for (auto *obj : to_remove)
    nvds_remove_obj_meta_from_frame (frame_meta, obj);
  nvds_release_meta_lock (batch_meta);

  for (guint i = 0; i < n; i++) {
    if (dets[i].merged_mask_data) {
      g_free (dets[i].merged_mask_data);
      dets[i].merged_mask_data = nullptr;
    }
  }
}

/* ── GstBaseTransform vmethod ────────────────────────────────────────────── */

static GstFlowReturn
gst_nvsahipostprocess_transform_ip (GstBaseTransform * btrans,
    GstBuffer * inbuf)
{
  GstNvSahiPostProcess *self = GST_NVSAHIPOSTPROCESS (btrans);

  GST_DEBUG_OBJECT (self, "transform_ip: buffer %p", inbuf);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (!batch_meta) {
    GST_DEBUG_OBJECT (self, "transform_ip: no batch_meta, passthrough");
    return GST_FLOW_OK;
  }

  std::vector<NvDsFrameMeta *> frames;
  for (NvDsMetaList *l = batch_meta->frame_meta_list; l; l = l->next)
    frames.push_back ((NvDsFrameMeta *) l->data);

  GST_DEBUG_OBJECT (self, "transform_ip: %zu frames in batch",
      frames.size ());

  auto t0 = std::chrono::steady_clock::now ();

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic) if(frames.size() > 1)
#endif
  for (size_t f = 0; f < frames.size (); f++)
    process_frame (self, batch_meta, frames[f]);

  if (G_UNLIKELY (GST_LEVEL_INFO <=
      gst_debug_category_get_threshold (GST_CAT_DEFAULT))) {
    auto t1 = std::chrono::steady_clock::now ();
    self->perf_accum_ms +=
        std::chrono::duration<gdouble, std::milli> (t1 - t0).count ();
    self->perf_batch_count++;
    self->perf_frame_count += frames.size ();
    gint64 now = g_get_monotonic_time ();
    if (now - self->perf_last_print_us >= 1000000) {
      GST_INFO_OBJECT (self,
          "PERF %.1fs: %u batches, %u frames | avg %.3f ms/batch, "
          "%.3f ms/frame | total %.1f ms",
          (now - self->perf_last_print_us) / 1e6,
          self->perf_batch_count, self->perf_frame_count,
          self->perf_accum_ms / self->perf_batch_count,
          self->perf_accum_ms / self->perf_frame_count,
          self->perf_accum_ms);
      self->perf_accum_ms = 0; self->perf_batch_count = 0;
      self->perf_frame_count = 0; self->perf_last_print_us = now;
    }
  }

  return GST_FLOW_OK;
}

/* ── Class / instance init ───────────────────────────────────────────────── */

static void
gst_nvsahipostprocess_class_init (GstNvSahiPostProcessClass * klass)
{
  GObjectClass *go = (GObjectClass *) klass;
  GstElementClass *ge = (GstElementClass *) klass;
  GstBaseTransformClass *bt = (GstBaseTransformClass *) klass;

  go->set_property = GST_DEBUG_FUNCPTR (gst_nvsahipostprocess_set_property);
  go->get_property = GST_DEBUG_FUNCPTR (gst_nvsahipostprocess_get_property);
  go->finalize     = GST_DEBUG_FUNCPTR (gst_nvsahipostprocess_finalize);
  bt->transform_ip = GST_DEBUG_FUNCPTR (gst_nvsahipostprocess_transform_ip);

#define RW (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
  g_object_class_install_property (go, PROP_GIE_IDS,
      g_param_spec_string ("gie-ids", "GIE IDs",
          "\"-1\"=all, or semicolon-separated ids (\"1;3;5\")", DEFAULT_GIE_IDS, RW));
  g_object_class_install_property (go, PROP_MATCH_METRIC,
      g_param_spec_uint ("match-metric", "Match Metric",
          "0=IoU, 1=IoS (recommended)", 0, 1, DEFAULT_MATCH_METRIC, RW));
  g_object_class_install_property (go, PROP_MATCH_THRESHOLD,
      g_param_spec_float ("match-threshold", "Match Threshold",
          "Overlap threshold for duplicate detection", 0.0f, 1.0f, DEFAULT_MATCH_THRESHOLD, RW));
  g_object_class_install_property (go, PROP_CLASS_AGNOSTIC,
      g_param_spec_boolean ("class-agnostic", "Class Agnostic",
          "Match across different class IDs", DEFAULT_CLASS_AGNOSTIC, RW));
  g_object_class_install_property (go, PROP_ENABLE_MERGE,
      g_param_spec_boolean ("enable-merge", "Enable Merge",
          "Merge suppressed boxes (NMM). FALSE=NMS suppress-only", DEFAULT_ENABLE_MERGE, RW));
  g_object_class_install_property (go, PROP_TWO_PHASE_NMM,
      g_param_spec_boolean ("two-phase-nmm", "Two-Phase NMM",
          "Phase 1: original bboxes, phase 2: re-check expanded", DEFAULT_TWO_PHASE_NMM, RW));
  g_object_class_install_property (go, PROP_MERGE_STRATEGY,
      g_param_spec_uint ("merge-strategy", "Merge Strategy",
          "0=union, 1=weighted, 2=largest", 0, 2, DEFAULT_MERGE_STRATEGY, RW));
  g_object_class_install_property (go, PROP_MAX_DETECTIONS,
      g_param_spec_int ("max-detections", "Max Detections",
          "Max surviving per frame (-1=unlimited)", -1, G_MAXINT, DEFAULT_MAX_DETECTIONS, RW));
  g_object_class_install_property (go, PROP_DROP_MASK_ON_MERGE,
      g_param_spec_boolean ("drop-mask-on-merge", "Drop Mask on Merge",
          "Clear mask on merge. FALSE=composite via max", DEFAULT_DROP_MASK_ON_MERGE, RW));

  gst_element_class_add_pad_template (ge, gst_static_pad_template_get (&src_tmpl));
  gst_element_class_add_pad_template (ge, gst_static_pad_template_get (&sink_tmpl));

  gst_element_class_set_details_simple (ge,
      "SAHI Post-Process (GreedyNMM v1.2)",
      "Filter/Metadata",
      "Merges duplicate detections from sliced inference using GreedyNMM "
      "with spatial indexing, mask merge, and parallel frame processing",
      "Levi Pereira <levi.pereira@gmail.com>");
}

static void
gst_nvsahipostprocess_init (GstNvSahiPostProcess * self)
{
  GstBaseTransform *bt = GST_BASE_TRANSFORM (self);
  gst_base_transform_set_in_place (bt, TRUE);
  gst_base_transform_set_passthrough (bt, FALSE);

  GST_DEBUG_OBJECT (self, "init: in_place=TRUE passthrough=FALSE");

  self->gie_ids_str         = nullptr;
  self->gie_ids             = new std::unordered_set<gint> ();
  self->gie_filter_all      = TRUE;
  parse_gie_ids (self, DEFAULT_GIE_IDS);
  self->match_metric        = DEFAULT_MATCH_METRIC;
  self->match_threshold     = DEFAULT_MATCH_THRESHOLD;
  self->class_agnostic      = DEFAULT_CLASS_AGNOSTIC;
  self->enable_merge        = DEFAULT_ENABLE_MERGE;
  self->two_phase_nmm       = DEFAULT_TWO_PHASE_NMM;
  self->merge_strategy      = DEFAULT_MERGE_STRATEGY;
  self->max_detections      = DEFAULT_MAX_DETECTIONS;
  self->drop_mask_on_merge  = DEFAULT_DROP_MASK_ON_MERGE;

  self->perf_accum_ms      = 0;
  self->perf_batch_count   = 0;
  self->perf_frame_count   = 0;
  self->perf_last_print_us = g_get_monotonic_time ();

  GST_DEBUG_OBJECT (self,
      "config: gie_ids=%s metric=%u threshold=%.2f agnostic=%d merge=%d "
      "two_phase=%d strategy=%u max_det=%d drop_mask=%d",
      self->gie_ids_str, self->match_metric, self->match_threshold,
      self->class_agnostic, self->enable_merge, self->two_phase_nmm,
      self->merge_strategy, self->max_detections, self->drop_mask_on_merge);
}

/* ── Cleanup ─────────────────────────────────────────────────────────────── */

static void
gst_nvsahipostprocess_finalize (GObject * object)
{
  GstNvSahiPostProcess *self = GST_NVSAHIPOSTPROCESS (object);
  g_free (self->gie_ids_str);
  self->gie_ids_str = nullptr;
  delete self->gie_ids;
  self->gie_ids = nullptr;
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* ── Property accessors ──────────────────────────────────────────────────── */

static void
gst_nvsahipostprocess_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvSahiPostProcess *s = GST_NVSAHIPOSTPROCESS (object);
  switch (prop_id) {
    case PROP_GIE_IDS:           parse_gie_ids (s, g_value_get_string (value)); break;
    case PROP_MATCH_METRIC:      s->match_metric = g_value_get_uint (value); break;
    case PROP_MATCH_THRESHOLD:   s->match_threshold = g_value_get_float (value); break;
    case PROP_CLASS_AGNOSTIC:    s->class_agnostic = g_value_get_boolean (value); break;
    case PROP_ENABLE_MERGE:      s->enable_merge = g_value_get_boolean (value); break;
    case PROP_TWO_PHASE_NMM:     s->two_phase_nmm = g_value_get_boolean (value); break;
    case PROP_MERGE_STRATEGY:    s->merge_strategy = g_value_get_uint (value); break;
    case PROP_MAX_DETECTIONS:    s->max_detections = g_value_get_int (value); break;
    case PROP_DROP_MASK_ON_MERGE:s->drop_mask_on_merge = g_value_get_boolean (value); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec); break;
  }
}

static void
gst_nvsahipostprocess_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvSahiPostProcess *s = GST_NVSAHIPOSTPROCESS (object);
  switch (prop_id) {
    case PROP_GIE_IDS:           g_value_set_string (value, s->gie_ids_str); break;
    case PROP_MATCH_METRIC:      g_value_set_uint (value, s->match_metric); break;
    case PROP_MATCH_THRESHOLD:   g_value_set_float (value, s->match_threshold); break;
    case PROP_CLASS_AGNOSTIC:    g_value_set_boolean (value, s->class_agnostic); break;
    case PROP_ENABLE_MERGE:      g_value_set_boolean (value, s->enable_merge); break;
    case PROP_TWO_PHASE_NMM:     g_value_set_boolean (value, s->two_phase_nmm); break;
    case PROP_MERGE_STRATEGY:    g_value_set_uint (value, s->merge_strategy); break;
    case PROP_MAX_DETECTIONS:    g_value_set_int (value, s->max_detections); break;
    case PROP_DROP_MASK_ON_MERGE:g_value_set_boolean (value, s->drop_mask_on_merge); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec); break;
  }
}

/* ── Plugin registration ─────────────────────────────────────────────────── */

static gboolean
nvsahipostprocess_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvsahipostprocess_debug,
      "nvsahipostprocess", 0, "SAHI post-process (GreedyNMM) plugin");

  return gst_element_register (plugin, "nvsahipostprocess",
      GST_RANK_PRIMARY, GST_TYPE_NVSAHIPOSTPROCESS);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_sahipostprocess,
    DESCRIPTION,
    nvsahipostprocess_plugin_init,
    VERSION,
    LICENSE,
    BINARY_PACKAGE,
    URL)
