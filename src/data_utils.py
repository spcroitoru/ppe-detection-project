"""
Utilities for working with the YOLO dataset (images + labels).
Functions extracted and consolidated from the EDA notebook (01_eda.ipynb),
now reusable across training/cleaning scripts.
"""
import glob
import os

from src.config import DATA_ROOT

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.jfif")


def list_images(split: str) -> list[str]:
    """Return the paths of all images in a given split (train/val/test)."""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(DATA_ROOT, split, "images", ext)))
    return files


def list_labels(split: str) -> list[str]:
    """Return the paths of all label files in a given split."""
    return glob.glob(os.path.join(DATA_ROOT, split, "labels", "*.txt"))


def clean_labels(split: str) -> dict:
    """
    Clean the labels of a split:
    - removes orphan labels (no matching image)
    - creates empty labels for images missing a label (treated as negative samples)
    Returns a report with the number of changes made.
    """
    img_dir = os.path.join(DATA_ROOT, split, "images")
    lbl_dir = os.path.join(DATA_ROOT, split, "labels")

    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in list_images(split)}
    lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in list_labels(split)}

    orphan_labels = lbl_stems - img_stems
    missing_labels = img_stems - lbl_stems

    removed = 0
    for stem in orphan_labels:
        os.remove(os.path.join(lbl_dir, f"{stem}.txt"))
        removed += 1

    created = 0
    for stem in missing_labels:
        open(os.path.join(lbl_dir, f"{stem}.txt"), "w").close()
        created += 1

    # remove the YOLO cache so it doesn't reuse stale data after cleaning
    cache_file = os.path.join(lbl_dir, "labels.cache")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    return {
        "split": split,
        "orphan_labels_removed": removed,
        "empty_labels_created": created,
    }


def check_data_quality(split: str) -> dict:
    """Check the structural quality of a split's data, without modifying anything."""
    img_basenames = {os.path.splitext(os.path.basename(p))[0] for p in list_images(split)}
    label_files = list_labels(split)
    label_basenames = {os.path.splitext(os.path.basename(p))[0] for p in label_files}

    images_without_label = img_basenames - label_basenames
    labels_without_image = label_basenames - img_basenames

    empty_labels = 0
    invalid_boxes = []
    for label_path in label_files:
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == 0:
            empty_labels += 1
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                invalid_boxes.append((label_path, "incomplete line"))
                continue
            _, xc, yc, bw, bh = parts[:5]
            xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                invalid_boxes.append((label_path, "coordinates outside [0,1]"))
            if bw <= 0 or bh <= 0:
                invalid_boxes.append((label_path, "zero width/height"))

    return {
        "split": split,
        "images_without_label": len(images_without_label),
        "labels_without_image": len(labels_without_image),
        "empty_labels": empty_labels,
        "invalid_bounding_boxes": len(invalid_boxes),
    }