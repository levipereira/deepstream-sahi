/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Levi Pereira <levi.pereira@gmail.com>
 * SPDX-License-Identifier: LicenseRef-NvidiaDeepStreamEULA
 *
 * 2D spatial hash grid for fast neighbor queries on axis-aligned bounding boxes.
 *
 * Divides the frame into uniform cells and maps each detection to the cells it
 * overlaps. Querying a bbox returns only detections in overlapping cells,
 * reducing average pair-check complexity from O(n^2) to roughly O(n * k) where
 * k is the average number of detections per cell neighborhood.
 */

#ifndef __SAHI_SPATIAL_GRID_H__
#define __SAHI_SPATIAL_GRID_H__

#include <vector>
#include <cmath>
#include <algorithm>
#include <glib.h>

struct SahiGridRect {
  gfloat left, top, right, bottom;
};

class SahiSpatialGrid {
public:
  void build (const std::vector<SahiGridRect> &boxes, guint count,
              gfloat frame_w, gfloat frame_h)
  {
    if (count == 0) return;

    gfloat max_dim = 0.0f;
    for (guint i = 0; i < count; i++) {
      gfloat w = boxes[i].right - boxes[i].left;
      gfloat h = boxes[i].bottom - boxes[i].top;
      max_dim = std::max (max_dim, std::max (w, h));
    }

    cell_size_ = std::max (max_dim, 1.0f);
    cols_ = static_cast<guint> (std::ceil (frame_w / cell_size_)) + 1;
    rows_ = static_cast<guint> (std::ceil (frame_h / cell_size_)) + 1;

    guint total_cells = cols_ * rows_;
    cells_.clear ();
    cells_.resize (total_cells);

    for (guint i = 0; i < count; i++) {
      guint c0, r0, c1, r1;
      cell_range (boxes[i], c0, r0, c1, r1);
      for (guint r = r0; r <= r1; r++) {
        for (guint c = c0; c <= c1; c++) {
          cells_[r * cols_ + c].push_back (i);
        }
      }
    }
  }

  void query (const SahiGridRect &box, std::vector<guint> &result) const
  {
    result.clear ();
    guint c0, r0, c1, r1;
    cell_range (box, c0, r0, c1, r1);
    for (guint r = r0; r <= r1; r++) {
      for (guint c = c0; c <= c1; c++) {
        const auto &cell = cells_[r * cols_ + c];
        result.insert (result.end (), cell.begin (), cell.end ());
      }
    }
    std::sort (result.begin (), result.end ());
    result.erase (std::unique (result.begin (), result.end ()), result.end ());
  }

  void clear ()
  {
    cells_.clear ();
    cols_ = rows_ = 0;
  }

private:
  void cell_range (const SahiGridRect &box,
                   guint &c0, guint &r0, guint &c1, guint &r1) const
  {
    c0 = static_cast<guint> (std::max (0.0f, box.left / cell_size_));
    r0 = static_cast<guint> (std::max (0.0f, box.top / cell_size_));
    c1 = std::min (static_cast<guint> (box.right / cell_size_), cols_ - 1);
    r1 = std::min (static_cast<guint> (box.bottom / cell_size_), rows_ - 1);
  }

  gfloat cell_size_ = 1.0f;
  guint  cols_ = 0;
  guint  rows_ = 0;
  std::vector<std::vector<guint>> cells_;
};

#endif /* __SAHI_SPATIAL_GRID_H__ */
