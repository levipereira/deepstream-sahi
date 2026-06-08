# Dataset — VisDrone, sliced into 416×416 tiles

This document specifies the dataset used to train the models in this bundle, ships the **exact
scripts** used to build it (`01_dataset/pipeline/`), and explains how to **distribute it via Google
Drive** so anyone can download the exact tiles and re-run the experiments.

## 1. Source

- **VisDrone2019** (drone-captured aerial detection). Both the **DET** (still images) and **VID**
  (video sequences) subsets were used — see the conversion script, which walks
  `VisDrone2019-DET-{train,val}` *and* `VisDrone2019-VID-{train,val,test-dev,test-challenge}`. This is
  why the tile counts are large (548k train tiles).
- Aerial frames are large (e.g. 1904×1071, 2688×1512, 1920×1080) and dominated by **tiny objects**,
  which is why a single 416/640 resize destroys recall. The remedy is **slicing** (tiling): cut each
  frame into overlapping windows so small objects occupy a usable fraction of the tile.
- Original raw VisDrone images live under `/dataset/visdrone/original/` (per the conversion script's
  default `--source`).

## 2. Slicing specification

| Property | Value |
|----------|-------|
| Tile size | **416 × 416** |
| Overlap | **20%** on both axes (step = 416 − 83 = **333 px**, i.e. 83 px overlap) |
| Keep rule | only tiles with **≥ 1 object** are saved (`--min-objects 1`) |
| Edge tiles | the partial crop is **resized up to 416×416** (no black padding) |
| Classes (`nc`) | **11** |
| Class names (id order) | `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, others` |
| Train tiles | **548,186** |
| Val tiles | **60,909** |
| Train/val split | last 1/5 of the produced files → val (80/20) |

### Actual on-disk composition (verified)

The final dataset is **not** 100% 416×416 tiles. Counting image dimensions across the full set:

| Split | Total images | 416×416 tiles | Full-resolution frames | Distinct sizes |
|-------|-------------:|--------------:|-----------------------:|---------------:|
| train | 548,186 | 510,148 (**93.1%**) | 38,038 (6.9%) | 16 |
| val   | 60,909  | 56,636 (**93.0%**)  | 4,273 (7.0%)  | 16 |

The ~7% non-tile images are **full-resolution original frames** at native VisDrone resolutions
(1904×1071, 2688×1512 for DET; 1920×1080, 960×540 for VID). The single slice script shipped here
(`02_create_visdrone_slice.py`) only ever emits 416×416 crops, so these full frames were retained
alongside the tiles by an additional pass when the on-disk set was assembled. They are part of the
dataset all models trained on, so they are kept (and documented) for exact reproducibility rather
than filtered out. Box annotations are clipped to tile boundaries and degenerate remnants dropped.

> Note: `wc -l train.txt / val.txt` reports 548,185 / 60,908 — one fewer each — only because the
> last line has no trailing newline. The split-list line counts are the authoritative
> 548,186 / 60,909.

## 3. On-disk format

The dataset is stored in **YOLO format** under:

| Path | Contents |
|------|----------|
| `/dataset/dfine/dataset/yolo_slice/` | One `.jpg` + one `.txt` per image (YOLO label format) |
| `/dataset/dfine/dataset/train.txt` | Absolute paths to all training images (548,186 lines) |
| `/dataset/dfine/dataset/val.txt` | Absolute paths to all validation images (60,909 lines) |

```
yolo_slice/
├── 00000000.jpg
├── 00000000.txt
├── 00000001.jpg
├── 00000001.txt
└── ...
+ /dataset/dfine/dataset/{train,val}.txt  (split lists)
```

`01_dataset/visdrone_slice.yaml` in this bundle is the Ultralytics data config (points at `train.txt` /
`val.txt`).

## 4. Reproducing the dataset from raw VisDrone (exact scripts included)

The two scripts that actually built this dataset are shipped in
**`01_dataset/pipeline/`** (copied verbatim from the machine that produced the benchmark):

| Step | Script | What it does |
|-----:|--------|--------------|
| 1 | `01_create_visdrone_yolo.py` | Raw VisDrone (DET + VID) → flat YOLO format. Maps VisDrone categories 1–11 → class ids 0–10, **drops category 0 (ignored regions)**, clips boxes to image bounds. |
| 2 | `02_create_visdrone_slice.py` | YOLO images → 416×416 tiles, 20% overlap, keep tiles with ≥1 object, resize edge crops to 416. Writes `train.txt`/`val.txt` (80/20) + a `visdrone_slice.yaml`. |

### Exact commands

```bash
# Step 1 — raw VisDrone -> YOLO  (defaults: --source /dataset/visdrone/original --output /dataset/visdrone_yolo)
python 01_dataset/pipeline/01_create_visdrone_yolo.py \
    --source /dataset/visdrone/original \
    --output /dataset/visdrone_yolo

# Step 2 — YOLO -> 416 tiles (20% overlap, min 1 object per tile)
python 01_dataset/pipeline/02_create_visdrone_slice.py \
    --source /dataset/visdrone_yolo \
    --output /dataset/visdrone_yolo_slice \
    --slice-size 416 --overlap-pct 20 --min-objects 1
```

> Keep the **11-class order above** so class ids match the trained weights.

### Key parameters baked into the scripts (for the record)

- `02_create_visdrone_slice.py`: `slice_size=416`, `overlap_pct=20.0` → `overlap = int(416*0.20)=83 px`,
  `step = 416-83 = 333`, `min_objects=1`, sliding window over `(y1,x1)` in steps of `step`, edge crops
  `cv2.resize(...,(416,416), INTER_LINEAR)`, val = last `len//5` of the file list.
- `01_create_visdrone_yolo.py`: VisDrone DET line `x,y,w,h,score,category,trunc,occ`; VID line
  `frame,object_id,x,y,w,h,score,category,trunc,occ`; `category==0` skipped; `class_id = category-1`.

## 5. Distribution — Google Drive

The sliced dataset is distributed via **Google Drive** (it is large — ~609k images — so it is not
committed to git). Download folder:

```
https://drive.google.com/drive/folders/1FUm51fN9N-pfciDwPr1s30_Ajy8wHGhR?usp=sharing
```

A single archive contains all images, YOLO labels, and split lists:

```bash
# Package the dataset
tar -C /dataset/dfine/dataset -czf visdrone_yolo_slice_416.tar.gz yolo_slice train.txt val.txt
```

Download & extract:

```bash
pip install gdown
# Fetch the whole shared folder...
gdown --folder "https://drive.google.com/drive/folders/1FUm51fN9N-pfciDwPr1s30_Ajy8wHGhR"
tar -xzf visdrone_yolo_slice_416.tar.gz      # -> yolo_slice/ + train.txt + val.txt
```

Then set `path:` in `01_dataset/visdrone_slice.yaml` to the extracted `yolo_slice/` directory.

### Optional alternative — Roboflow
A helper (`01_dataset/roboflow_upload.py`, reads the key from `$ROBOFLOW_API_KEY`) can publish the
dataset to Roboflow instead, if you prefer a hosted/versioned dataset. Note ~609k images exceeds the
free tier; generate the version with **no extra augmentation / no resize** (tiles are already
416×416) and export as `YOLOv8`/`YOLO11` (Ultralytics).

## 6. License / attribution
VisDrone is released for **academic, non-commercial** use. Keep the VisDrone citation and license with
any redistribution (Google Drive, Roboflow, etc.), and mark the derived tiles as a derivative of
VisDrone2019.
