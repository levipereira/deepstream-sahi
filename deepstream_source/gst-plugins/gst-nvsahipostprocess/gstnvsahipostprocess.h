/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Levi Pereira <levi.pereira@gmail.com>
 * SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
 *
 * Subject to the NVIDIA DeepStream SDK License Agreement:
 * https://developer.nvidia.com/deepstream-eula
 *
 * DeepStream SAHI Post-Process Plugin
 * Merges duplicate detections from sliced inference using GreedyNMM with
 * spatial indexing, two-phase merge, per-class partitioning, mask merge,
 * and parallel per-frame processing.
 */

#ifndef __GST_NVSAHIPOSTPROCESS_H__
#define __GST_NVSAHIPOSTPROCESS_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "gstnvdsmeta.h"
#include "spatial_grid.h"
#include "mask_merge.h"

#define PACKAGE "nvsahipostprocess"
#define VERSION "1.2"
#define LICENSE "Proprietary"
#define DESCRIPTION \
    "DeepStream SAHI post-process plugin — GreedyNMM duplicate merging " \
    "with spatial indexing, mask merge, and parallel frame processing"
#define BINARY_PACKAGE "DeepStream SAHI Post-Process"
#define URL "https://github.com/levipereira/deepstream-sahi"

G_BEGIN_DECLS

typedef struct _GstNvSahiPostProcess GstNvSahiPostProcess;
typedef struct _GstNvSahiPostProcessClass GstNvSahiPostProcessClass;

#define GST_TYPE_NVSAHIPOSTPROCESS (gst_nvsahipostprocess_get_type())
#define GST_NVSAHIPOSTPROCESS(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVSAHIPOSTPROCESS, GstNvSahiPostProcess))
#define GST_NVSAHIPOSTPROCESS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVSAHIPOSTPROCESS, GstNvSahiPostProcessClass))
#define GST_IS_NVSAHIPOSTPROCESS(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVSAHIPOSTPROCESS))
#define GST_IS_NVSAHIPOSTPROCESS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVSAHIPOSTPROCESS))

typedef enum {
  SAHI_METRIC_IOU = 0,
  SAHI_METRIC_IOS = 1,
} SahiMatchMetric;

typedef enum {
  SAHI_MERGE_UNION    = 0,
  SAHI_MERGE_WEIGHTED = 1,
  SAHI_MERGE_LARGEST  = 2,
} SahiMergeStrategy;

typedef struct {
  gfloat left, top, right, bottom;
  gfloat orig_left, orig_top, orig_right, orig_bottom;
  gfloat score;
  gint   class_id;
  gchar  obj_label[128];
  gfloat area;
  NvDsObjectMeta *obj_meta;
  gboolean merged;

  /* mask support */
  SahiMaskData mask;
  float *merged_mask_data;

  /* cross-class merge: track highest-scoring contributor */
  gfloat best_score;
  gint   best_class_id;
  gchar  best_label[128];
} SahiDetection;

struct _GstNvSahiPostProcess
{
  GstBaseTransform base_trans;

  /* properties */
  gchar            *gie_ids_str;
  std::unordered_set<gint> *gie_ids;
  gboolean          gie_filter_all;
  guint             match_metric;
  gfloat            match_threshold;
  gboolean          class_agnostic;
  gboolean          enable_merge;
  gboolean          two_phase_nmm;
  guint             merge_strategy;
  gint              max_detections;
  gboolean          drop_mask_on_merge;

  /* latency profiling (GST_DEBUG=nvsahipostprocess:4) */
  gdouble           perf_accum_ms;
  guint             perf_batch_count;
  guint             perf_frame_count;
  gint64            perf_last_print_us;
};

struct _GstNvSahiPostProcessClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvsahipostprocess_get_type (void);

G_END_DECLS

#endif /* __GST_NVSAHIPOSTPROCESS_H__ */
