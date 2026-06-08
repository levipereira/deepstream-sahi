#!/usr/bin/env python3
"""
Gera dataset de slices a partir de visdrone_yolo com 20% de overlap nos 2 eixos.

Para cada imagem:
  - Recorta em tiles (sliding window com overlap)
  - overlap = 20% do slice_size em ambos os eixos
  - Só salva slices com pelo menos 1 objeto
  - Redimensiona crop real (sem padding preto)

Saída: /dataset/visdrone_yolo_slice
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def xywhn2xyxy_np(lb, w, h):
    """lb: (n,4) - xc,yc,w,h normalizado -> x1,y1,x2,y2 pixels"""
    x = np.asarray(lb, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = np.zeros_like(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2)
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2)
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2)
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2)
    return y


def xyxy2xywhn_np(xyxy, w, h, eps=1e-3):
    """xyxy: (n,4) pixels -> xc,yc,w,h normalizado, clipado"""
    x = np.asarray(xyxy, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x[..., 0] = np.clip(x[..., 0], 0, w - eps)
    x[..., 1] = np.clip(x[..., 1], 0, h - eps)
    x[..., 2] = np.clip(x[..., 2], eps, w)
    x[..., 3] = np.clip(x[..., 3], eps, h)
    y = np.zeros_like(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h
    y[..., 2] = np.maximum((x[..., 2] - x[..., 0]) / w, eps)
    y[..., 3] = np.maximum((x[..., 3] - x[..., 1]) / h, eps)
    return y


def get_slices(h, w, slice_size=416, overlap=0):
    """Gera (x1, y1, x2, y2) para cada slice. overlap em pixels (20% = slice_size*0.2)."""
    step = max(1, slice_size - overlap)
    slices = []
    for y1 in range(0, h, step):
        for x1 in range(0, w, step):
            x2 = min(x1 + slice_size, w)
            y2 = min(y1 + slice_size, h)
            if x2 > x1 and y2 > y1:
                slices.append((x1, y1, x2, y2))
    return slices


def clip_boxes_to_slice(labels, img_w, img_h, x1, y1, slice_w, slice_h):
    """Labels (n,5) -> labels no slice com bboxes clippados."""
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    cls = labels[:, 0:1]
    xyxy = xywhn2xyxy_np(labels[:, 1:5], img_w, img_h)

    x1_c = np.maximum(xyxy[:, 0], x1)
    y1_c = np.maximum(xyxy[:, 1], y1)
    x2_c = np.minimum(xyxy[:, 2], x1 + slice_w)
    y2_c = np.minimum(xyxy[:, 3], y1 + slice_h)

    valid = (x2_c > x1_c + 1) & (y2_c > y1_c + 1)
    if not valid.any():
        return np.zeros((0, 5), dtype=np.float32)

    xyxy_rel = np.stack([x1_c[valid] - x1, y1_c[valid] - y1, x2_c[valid] - x1, y2_c[valid] - y1], axis=1)
    xywhn = xyxy2xywhn_np(xyxy_rel, slice_w, slice_h)
    return np.concatenate([cls[valid], xywhn], axis=1).astype(np.float32)


def process_image(img_path, label_path, output_dir, slice_size, overlap, min_objects, counter):
    """Processa uma imagem: gera slices e salva."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]

    labels = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    slices = get_slices(h, w, slice_size, overlap)
    saved = 0

    for x1, y1, x2, y2 in slices:
        crop = img[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        slice_labels = clip_boxes_to_slice(labels, w, h, x1, y1, x2 - x1, y2 - y1)

        if len(slice_labels) < min_objects:
            continue

        if crop_w != slice_size or crop_h != slice_size:
            crop = cv2.resize(crop, (slice_size, slice_size), interpolation=cv2.INTER_LINEAR)

        base = counter[0]
        counter[0] += 1
        out_img = output_dir / f"{base:08d}.jpg"
        out_lbl = output_dir / f"{base:08d}.txt"

        cv2.imwrite(str(out_img), crop)
        with open(out_lbl, "w") as f:
            for row in slice_labels:
                f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="Gera slices VisDrone com 20%% overlap")
    parser.add_argument("--source", default="/dataset/visdrone_yolo", help="Dataset YOLO VisDrone")
    parser.add_argument("--output", default="/dataset/visdrone_yolo_slice", help="Diretório de saída")
    parser.add_argument("--slice-size", type=int, default=416, help="Tamanho do slice")
    parser.add_argument("--overlap-pct", type=float, default=20.0,
                        help="Overlap em %% (20 = 20%% nos 2 eixos)")
    parser.add_argument("--min-objects", type=int, default=1,
                        help="Mínimo de objetos no slice para salvar")
    parser.add_argument("--train-txt", default=None, help="train.txt com paths")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    overlap = int(args.slice_size * args.overlap_pct / 100)

    if not source.exists():
        print(f"Erro: {source} não existe. Execute primeiro create_visdrone_yolo.py")
        return 1

    output.mkdir(parents=True, exist_ok=True)

    if args.train_txt:
        train_txt = Path(args.train_txt)
        if train_txt.exists():
            with open(train_txt) as f:
                img_paths = [Path(p.strip()) for p in f if p.strip()]
            val_txt = train_txt.parent / "val.txt"
            if val_txt.exists():
                with open(val_txt) as f:
                    img_paths += [Path(p.strip()) for p in f if p.strip()]
        else:
            img_paths = []
    else:
        img_paths = list(source.glob("*.jpg")) + list(source.glob("*.jpeg")) + list(source.glob("*.png"))
    img_paths = sorted(set(p for p in img_paths if p.exists() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}))

    if not img_paths:
        print(f"Erro: nenhuma imagem em {source}")
        return 1

    print(f"Processando {len(img_paths)} imagens -> {output}")
    print(f"Slice: {args.slice_size}x{args.slice_size}, overlap={args.overlap_pct}%% ({overlap}px), min_objects={args.min_objects}\n")

    counter = [0]
    total_slices = 0
    for img_path in tqdm(img_paths, desc="Slices"):
        label_path = img_path.with_suffix(".txt")
        n = process_image(img_path, label_path, output, args.slice_size, overlap, args.min_objects, counter)
        total_slices += n

    all_slices = sorted(output.glob("*.jpg")) + sorted(output.glob("*.jpeg")) + sorted(output.glob("*.png"))
    paths = [str(p) for p in all_slices]
    n_val = max(1, len(paths) // 5)
    train_paths = paths[:-n_val]
    val_paths = paths[-n_val:]

    for name, p in [("train.txt", train_paths), ("val.txt", val_paths)]:
        with open(output / name, "w") as f:
            f.write("\n".join(p) + "\n")
        print(f"{name}: {len(p)} slices")

    yaml_path = Path(__file__).parent / "data" / "visdrone_slice.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = f"""# VisDrone YOLO Slice {args.slice_size}x{args.slice_size} (20% overlap)
path: {output.resolve()}
train: train.txt
val: val.txt

nc: 11
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
  10: others
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nTotal: {total_slices} slices em {output}")
    print(f"Config: {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
