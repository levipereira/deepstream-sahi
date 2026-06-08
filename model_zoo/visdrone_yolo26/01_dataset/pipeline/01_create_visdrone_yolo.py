#!/usr/bin/env python3
"""
Converte o dataset VisDrone para formato YOLO com imagens e labels no mesmo diretório.

Entrada: /dataset/visdrone/original
  Processa TODOS os diretórios:
  - VisDrone2019-DET-train, DET-val (images/ + annotations/)
  - VisDrone2019-VID-train, VID-val, VID-test-dev, VID-test-challenge (sequences/ + annotations/)
  - images/ + annotations/ na raiz

Formato VisDrone DET: x,y,w,h,score,category,truncation,occlusion (top-left + w,h)
Formato VisDrone VID: frame,object_id,x,y,w,h,score,category,truncation,occlusion

  category: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 6=truck,
            7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others

Formato YOLO: class_id x_center y_center width height (normalizado 0-1)
"""

import argparse
import shutil
from pathlib import Path

import cv2

# VisDrone category 0 = ignored (skip). 1-11 mapeados para 0-10
VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor", "others"
]


def convert_visdrone_line(line: str, img_w: int, img_h: int, is_vid: bool = False):
    """
    VisDrone DET: x,y,w,h,score,category,truncation,occlusion
    VisDrone VID: frame,object_id,x,y,w,h,score,category,truncation,occlusion
    Retorna (class_id, xc, yc, w, h) YOLO ou None se ignorar.
    """
    parts = line.strip().split(",")
    if is_vid and len(parts) < 8:
        return None
    if not is_vid and len(parts) < 6:
        return None
    try:
        off = 2 if is_vid else 0  # VID tem frame,object_id no início
        x = int(parts[off + 0])
        y = int(parts[off + 1])
        w = int(parts[off + 2])
        h = int(parts[off + 3])
        category = int(parts[off + 5])
    except (ValueError, IndexError):
        return None

    if category == 0:  # ignored regions
        return None
    if category < 1 or category > 11:
        return None

    # Clip ao bounds da imagem
    x1 = max(0, min(x, img_w - 1))
    y1 = max(0, min(y, img_h - 1))
    x2 = max(x1 + 1, min(x + w, img_w))
    y2 = max(y1 + 1, min(y + h, img_h))
    w_clip = x2 - x1
    h_clip = y2 - y1
    if w_clip < 2 or h_clip < 2:
        return None

    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = w_clip / img_w
    height = h_clip / img_h

    x_center = max(0.001, min(0.999, x_center))
    y_center = max(0.001, min(0.999, y_center))
    width = max(0.001, min(1, width))
    height = max(0.001, min(1, height))

    class_id = category - 1  # 1-11 -> 0-10
    return (class_id, x_center, y_center, width, height)


def process_split(img_dir: Path, ann_dir: Path, output_dir: Path, copy_images: bool):
    """Processa um split (train ou val) e grava em output_dir. Retorna (conv, empty, n, paths)."""
    if not img_dir.exists() or not ann_dir.exists():
        return 0, 0, 0, []

    ext = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted(f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in ext)
    paths = []

    converted = 0
    empty = 0

    for img_path in img_files:
        ann_path = ann_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        labels = []
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    result = convert_visdrone_line(line, img_w, img_h)
                    if result is not None:
                        labels.append(result)

        out_img = output_dir / img_path.name
        out_label = output_dir / (img_path.stem + ".txt")

        if copy_images:
            shutil.copy2(img_path, out_img)
        else:
            if not out_img.exists():
                out_img.symlink_to(img_path.resolve())

        with open(out_label, "w") as f:
            for cls_id, xc, yc, w, h in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        paths.append(str(out_img))
        if labels:
            converted += 1
        else:
            empty += 1

    return converted, empty, len(img_files), paths


def process_vid_folder(vid_root: Path, output_dir: Path, copy_images: bool, split_name: str):
    """
    Processa pasta VID: sequences/ + annotations/
    annotations/uav0000013_00000_v.txt -> sequences/uav0000013_00000_v/*.jpg
    Formato VID: frame,object_id,x,y,w,h,score,category,truncation,occlusion
    """
    seq_dir = vid_root / "sequences"
    ann_dir = vid_root / "annotations"
    if not seq_dir.exists() or not ann_dir.exists():
        return 0, 0, 0, []

    ext = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    converted = 0
    empty = 0
    total = 0

    for seq_path in sorted(seq_dir.iterdir()):
        if not seq_path.is_dir():
            continue
        seq_name = seq_path.name
        ann_file = ann_dir / (seq_name + ".txt")
        if not ann_file.exists():
            continue

        # Carregar anotações por frame
        frame_anns = {}
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                try:
                    frame_id = int(parts[0])
                except ValueError:
                    continue
                if frame_id not in frame_anns:
                    frame_anns[frame_id] = []
                frame_anns[frame_id].append(line)

        for img_path in sorted(seq_path.iterdir()):
            if img_path.suffix.lower() not in ext:
                continue
            total += 1
            stem = img_path.stem
            try:
                frame_id = int(stem)
            except ValueError:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            labels = []
            for line in frame_anns.get(frame_id, []):
                result = convert_visdrone_line(line, img_w, img_h, is_vid=True)
                if result is not None:
                    labels.append(result)

            out_name = f"{seq_name}_{stem}.jpg"
            out_img = output_dir / out_name
            out_label = output_dir / (f"{seq_name}_{stem}.txt")

            if copy_images:
                shutil.copy2(img_path, out_img)
            else:
                if not out_img.exists():
                    out_img.symlink_to(img_path.resolve())

            with open(out_label, "w") as f:
                for cls_id, xc, yc, w, h in labels:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            paths.append(str(out_img))
            if labels:
                converted += 1
            else:
                empty += 1

    return converted, empty, total, paths


def main():
    parser = argparse.ArgumentParser(
        description="Converte VisDrone para YOLO (imagem + label no mesmo diretório)"
    )
    parser.add_argument("--source", default="/dataset/visdrone/original", help="Diretório VisDrone original")
    parser.add_argument("--output", default="/dataset/visdrone_yolo", help="Diretório de saída")
    parser.add_argument("--no-copy", action="store_true", help="Usar symlinks em vez de copiar")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        print(f"Erro: {source} não existe")
        return 1

    output.mkdir(parents=True, exist_ok=True)
    copy_images = not args.no_copy

    print(f"Convertendo VisDrone: {source} -> {output}")
    print(f"Modo: {'symlinks' if args.no_copy else 'cópia'}\n")

    total_conv = 0
    total_empty = 0
    total_imgs = 0
    train_paths = []
    val_paths = []

    # DET-train e DET-val
    for folder, split_name in [("VisDrone2019-DET-train", "train"), ("VisDrone2019-DET-val", "val")]:
        img_dir = source / folder / "images"
        ann_dir = source / folder / "annotations"
        conv, empty, n, paths = process_split(img_dir, ann_dir, output, copy_images)
        total_conv += conv
        total_empty += empty
        total_imgs += n
        if split_name == "train":
            train_paths.extend(paths)
        else:
            val_paths.extend(paths)
        if n > 0:
            print(f"  {folder}: {n} imagens, {conv} com anotações, {empty} vazias")

    # VID-train, VID-val, VID-test-dev, VID-test-challenge
    for folder in ["VisDrone2019-VID-train", "VisDrone2019-VID-val", "VisDrone2019-VID-test-dev", "VisDrone2019-VID-test-challenge"]:
        vid_root = source / folder
        if not vid_root.exists():
            continue
        conv, empty, n, paths = process_vid_folder(vid_root, output, copy_images, folder)
        total_conv += conv
        total_empty += empty
        total_imgs += n
        if n > 0:
            is_val = "val" in folder or "test" in folder
            if is_val:
                val_paths.extend(paths)
            else:
                train_paths.extend(paths)
            print(f"  {folder}: {n} imagens, {conv} com anotações, {empty} vazias")

    # images/ e annotations/ na raiz
    img_dir = source / "images"
    ann_dir = source / "annotations"
    if img_dir.exists() and ann_dir.exists():
        conv, empty, n, paths = process_split(img_dir, ann_dir, output, copy_images)
        total_conv += conv
        total_empty += empty
        total_imgs += n
        if n > 0:
            train_paths.extend(paths)
            print(f"  images/ (raiz): {n} imagens, {conv} com anotações, {empty} vazias")

    # Deduplicar (mesmo path pode vir de múltiplas fontes)
    train_paths = sorted(set(train_paths))
    val_paths = sorted(set(val_paths))
    if not val_paths and train_paths:
        n_val = max(1, len(train_paths) // 5)
        val_paths = train_paths[-n_val:]
        train_paths = train_paths[:-n_val]

    for name, paths in [("train.txt", train_paths), ("val.txt", val_paths)]:
        if paths:
            with open(output / name, "w") as f:
                f.write("\n".join(paths) + "\n")
            print(f"\nCriado {name} com {len(paths)} imagens")

    # data.yaml
    yaml_path = Path(__file__).parent / "data" / "visdrone.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    names_str = "\n".join(f"  {i}: {n}" for i, n in enumerate(VISDRONE_NAMES))
    yaml_content = f"""# VisDrone YOLO - imagens e labels no mesmo diretório
path: {output.resolve()}
train: train.txt
val: val.txt

nc: 11
names:
{names_str}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nConfig salvo em: {yaml_path}")
    print(f"\nTotal: {total_imgs} imagens, {total_conv} com labels")
    return 0


if __name__ == "__main__":
    exit(main())
