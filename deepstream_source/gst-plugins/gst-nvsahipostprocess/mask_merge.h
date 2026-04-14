/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Levi Pereira <levi.pereira@gmail.com>
 * SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
 *
 * Instance-segmentation mask merge utilities for the SAHI postprocess plugin.
 *
 * DeepStream masks (NvOSD_MaskParams) are dense float arrays sized
 * (width x height) covering the detection bounding box. When two detections
 * are merged, their masks must be composited into the expanded union bbox.
 *
 * Strategy:
 *   1. Allocate a new mask buffer covering the merged bbox dimensions.
 *   2. For each source mask, map its pixels into the merged coordinate space
 *      using nearest-neighbor resampling.
 *   3. Take the element-wise maximum of both projected masks.
 */

#ifndef __SAHI_MASK_MERGE_H__
#define __SAHI_MASK_MERGE_H__

#include <cstring>
#include <cmath>
#include <algorithm>
#include <glib.h>

struct SahiMaskData {
  float   *data;
  guint    width;
  guint    height;
  float    threshold;
  gboolean valid;
};

static inline SahiMaskData
sahi_mask_from_nvds (float *data, guint w, guint h, float thr)
{
  SahiMaskData m;
  m.data = data;
  m.width = w;
  m.height = h;
  m.threshold = thr;
  m.valid = (data != nullptr && w > 0 && h > 0);
  return m;
}

/*
 * Project src mask (covering src_box) onto a buffer (covering dst_box) using
 * nearest-neighbor sampling, taking element-wise max with existing dst values.
 *
 * Coordinates are in the same frame-level pixel space.
 */
static inline void
sahi_mask_project_max (const SahiMaskData &src,
                       gfloat src_left, gfloat src_top,
                       gfloat src_right, gfloat src_bottom,
                       float *dst, guint dst_w, guint dst_h,
                       gfloat dst_left, gfloat dst_top,
                       gfloat dst_right, gfloat dst_bottom)
{
  if (!src.valid || !dst)
    return;

  gfloat dst_box_w = dst_right - dst_left;
  gfloat dst_box_h = dst_bottom - dst_top;
  gfloat src_box_w = src_right - src_left;
  gfloat src_box_h = src_bottom - src_top;

  if (dst_box_w <= 0 || dst_box_h <= 0 || src_box_w <= 0 || src_box_h <= 0)
    return;

  for (guint dy = 0; dy < dst_h; dy++) {
    gfloat frame_y = dst_top + (dy + 0.5f) * dst_box_h / dst_h;
    gfloat src_fy = (frame_y - src_top) / src_box_h * src.height;
    gint sy = static_cast<gint> (src_fy);
    if (sy < 0 || sy >= (gint) src.height)
      continue;

    for (guint dx = 0; dx < dst_w; dx++) {
      gfloat frame_x = dst_left + (dx + 0.5f) * dst_box_w / dst_w;
      gfloat src_fx = (frame_x - src_left) / src_box_w * src.width;
      gint sx = static_cast<gint> (src_fx);
      if (sx < 0 || sx >= (gint) src.width)
        continue;

      guint si = sy * src.width + sx;
      guint di = dy * dst_w + dx;
      dst[di] = std::max (dst[di], src.data[si]);
    }
  }
}

/*
 * Merge two masks into a newly allocated buffer covering the union bbox.
 * Returns the merged mask data; caller owns the returned data pointer.
 * Output dimensions use the larger of the two source mask resolutions scaled
 * proportionally to the union bbox.
 */
static inline SahiMaskData
sahi_mask_merge (const SahiMaskData &m1,
                 gfloat l1, gfloat t1, gfloat r1, gfloat b1,
                 const SahiMaskData &m2,
                 gfloat l2, gfloat t2, gfloat r2, gfloat b2,
                 gfloat ml, gfloat mt, gfloat mr, gfloat mb)
{
  SahiMaskData out = {nullptr, 0, 0, 0.5f, FALSE};

  gfloat mw = mr - ml;
  gfloat mh = mb - mt;
  if (mw <= 0 || mh <= 0)
    return out;

  gfloat px_per_unit = 0.0f;
  if (m1.valid) {
    gfloat bw1 = r1 - l1;
    if (bw1 > 0) px_per_unit = std::max (px_per_unit, m1.width / bw1);
  }
  if (m2.valid) {
    gfloat bw2 = r2 - l2;
    if (bw2 > 0) px_per_unit = std::max (px_per_unit, m2.width / bw2);
  }
  if (px_per_unit <= 0)
    return out;

  out.width = std::max (1u, static_cast<guint> (mw * px_per_unit + 0.5f));
  out.height = std::max (1u, static_cast<guint> (mh * px_per_unit + 0.5f));

  const guint cap = 512;
  out.width = std::min (out.width, cap);
  out.height = std::min (out.height, cap);

  guint buf_size = out.width * out.height;
  out.data = static_cast<float *> (g_malloc0 (buf_size * sizeof (float)));
  out.threshold = m1.valid ? m1.threshold : m2.threshold;
  out.valid = TRUE;

  if (m1.valid)
    sahi_mask_project_max (m1, l1, t1, r1, b1,
                           out.data, out.width, out.height, ml, mt, mr, mb);
  if (m2.valid)
    sahi_mask_project_max (m2, l2, t2, r2, b2,
                           out.data, out.width, out.height, ml, mt, mr, mb);

  return out;
}

#endif /* __SAHI_MASK_MERGE_H__ */
