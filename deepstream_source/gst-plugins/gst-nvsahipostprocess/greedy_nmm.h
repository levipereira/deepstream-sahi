/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Levi Pereira <levi.pereira@gmail.com>
 * SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
 *
 * GreedyNMM algorithm core: overlap computation, merge helpers, deterministic
 * sorting, and the two-phase NMM loop with spatial grid acceleration.
 */

#ifndef __SAHI_GREEDY_NMM_H__
#define __SAHI_GREEDY_NMM_H__

#include <algorithm>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <cstring>
#include "gstnvsahipostprocess.h"

/* ── Overlap computation ─────────────────────────────────────────────────── */

static inline gfloat
compute_overlap (guint metric, const SahiDetection &a, const SahiDetection &b)
{
  gfloat il = MAX (a.left, b.left);
  gfloat it = MAX (a.top, b.top);
  gfloat ir = MIN (a.right, b.right);
  gfloat ib = MIN (a.bottom, b.bottom);

  if (ir <= il || ib <= it)
    return 0.0f;

  gfloat inter = (ir - il) * (ib - it);

  if (metric == SAHI_METRIC_IOU) {
    gfloat u = a.area + b.area - inter;
    return (u > 0.0f) ? inter / u : 0.0f;
  } else {
    gfloat s = MIN (a.area, b.area);
    return (s > 0.0f) ? inter / s : 0.0f;
  }
}

static inline gfloat
compute_overlap_orig (guint metric,
    const SahiDetection &a, const SahiDetection &b)
{
  gfloat il = MAX (a.orig_left, b.orig_left);
  gfloat it = MAX (a.orig_top, b.orig_top);
  gfloat ir = MIN (a.orig_right, b.orig_right);
  gfloat ib = MIN (a.orig_bottom, b.orig_bottom);

  if (ir <= il || ib <= it)
    return 0.0f;

  gfloat inter = (ir - il) * (ib - it);
  gfloat a_area = (a.orig_right - a.orig_left) * (a.orig_bottom - a.orig_top);
  gfloat b_area = (b.orig_right - b.orig_left) * (b.orig_bottom - b.orig_top);

  if (metric == SAHI_METRIC_IOU) {
    gfloat u = a_area + b_area - inter;
    return (u > 0.0f) ? inter / u : 0.0f;
  } else {
    gfloat s = MIN (a_area, b_area);
    return (s > 0.0f) ? inter / s : 0.0f;
  }
}

/* ── Merge helpers ───────────────────────────────────────────────────────── */

static inline void
merge_bbox (SahiDetection &dst, const SahiDetection &src, guint strategy)
{
  if (strategy == SAHI_MERGE_WEIGHTED) {
    gfloat ws = dst.score + src.score;
    if (ws > 0.0f) {
      gfloat wd = dst.score / ws, wk = src.score / ws;
      dst.left   = dst.left   * wd + src.left   * wk;
      dst.top    = dst.top    * wd + src.top    * wk;
      dst.right  = dst.right  * wd + src.right  * wk;
      dst.bottom = dst.bottom * wd + src.bottom * wk;
    }
  } else if (strategy == SAHI_MERGE_LARGEST) {
    if (src.area > dst.area) {
      dst.left = src.left; dst.top = src.top;
      dst.right = src.right; dst.bottom = src.bottom;
    }
  } else {
    dst.left   = MIN (dst.left,   src.left);
    dst.top    = MIN (dst.top,    src.top);
    dst.right  = MAX (dst.right,  src.right);
    dst.bottom = MAX (dst.bottom, src.bottom);
  }

  if (src.score > dst.best_score) {
    dst.best_score = src.score;
    dst.best_class_id = src.class_id;
    memcpy (dst.best_label, src.obj_label, sizeof (dst.best_label));
  }

  dst.score = MAX (dst.score, src.score);
  dst.area = (dst.right - dst.left) * (dst.bottom - dst.top);
  dst.merged = TRUE;
}

/* ── Deterministic sort comparator ───────────────────────────────────────── */

static inline bool
det_compare (const std::vector<SahiDetection> &dets, guint a, guint b)
{
  if (dets[a].score != dets[b].score)
    return dets[a].score > dets[b].score;
  auto ca = std::tie (dets[a].left, dets[a].top, dets[a].right, dets[a].bottom);
  auto cb = std::tie (dets[b].left, dets[b].top, dets[b].right, dets[b].bottom);
  return ca < cb;
}

/* ── Two-phase GreedyNMM with spatial grid ───────────────────────────────── */

static inline void
run_nmm_on_group (
    std::vector<SahiDetection> &dets,
    std::vector<uint8_t> &suppressed,
    const SahiSpatialGrid &grid,
    const std::vector<guint> &idx_set,
    gfloat threshold, guint metric, guint strategy,
    gboolean agnostic, gboolean do_merge, gboolean two_phase,
    gboolean drop_mask_on_merge)
{
  std::vector<guint> candidates;
  std::unordered_map<guint, std::vector<guint>> merge_list;

  /* Phase 1: build merge lists using original coordinates */
  for (guint ii = 0; ii < idx_set.size (); ii++) {
    guint i = idx_set[ii];
    if (suppressed[i]) continue;

    grid.query ({dets[i].orig_left, dets[i].orig_top,
                 dets[i].orig_right, dets[i].orig_bottom}, candidates);

    std::vector<guint> my_merges;

    for (guint c : candidates) {
      if (c == i || suppressed[c]) continue;
      if (!agnostic && dets[i].class_id != dets[c].class_id) continue;
      if (det_compare (dets, c, i)) continue;

      gfloat ov = compute_overlap_orig (metric, dets[i], dets[c]);
      if (ov >= threshold) {
        suppressed[c] = 1;
        if (do_merge)
          my_merges.push_back (c);
      }
    }
    if (do_merge && !my_merges.empty ())
      merge_list[i] = std::move (my_merges);
  }

  if (!do_merge) return;

  /* Phase 2: re-check against expanding bbox, then merge */
  for (guint ii = 0; ii < idx_set.size (); ii++) {
    guint i = idx_set[ii];
    auto it = merge_list.find (i);
    if (it == merge_list.end ()) continue;

    for (guint j : it->second) {
      gfloat ov;
      if (two_phase)
        ov = compute_overlap (metric, dets[i], dets[j]);
      else
        ov = threshold;

      if (ov >= threshold) {
        gfloat pre_l = dets[i].left, pre_t = dets[i].top;
        gfloat pre_r = dets[i].right, pre_b = dets[i].bottom;
        merge_bbox (dets[i], dets[j], strategy);

        if (!drop_mask_on_merge &&
            (dets[i].mask.valid || dets[j].mask.valid)) {
          SahiMaskData mm = sahi_mask_merge (
              dets[i].merged_mask_data ?
                sahi_mask_from_nvds (dets[i].merged_mask_data,
                    dets[i].mask.width, dets[i].mask.height,
                    dets[i].mask.threshold) : dets[i].mask,
              pre_l, pre_t, pre_r, pre_b,
              dets[j].mask,
              dets[j].orig_left, dets[j].orig_top,
              dets[j].orig_right, dets[j].orig_bottom,
              dets[i].left, dets[i].top,
              dets[i].right, dets[i].bottom);
          if (mm.valid) {
            if (dets[i].merged_mask_data)
              g_free (dets[i].merged_mask_data);
            dets[i].merged_mask_data = mm.data;
            dets[i].mask.width = mm.width;
            dets[i].mask.height = mm.height;
            dets[i].mask.threshold = mm.threshold;
            dets[i].mask.valid = TRUE;
          }
        }
      }
    }
  }
}

#endif /* __SAHI_GREEDY_NMM_H__ */
