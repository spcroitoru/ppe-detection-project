"""
Investigation script: extracts and saves a few example images that contain
objects from the unknown class (id 10), with the bounding box drawn on them,
so we can visually identify what it represents.

Run from the project root (ppe-detection-project):
    python investigate_class10.py
"""
import glob
import os
import cv2

LABELS_DIR = "data_raw/train/labels"
IMAGES_DIR = "data_raw/train/images"
OUTPUT_DIR = "class10_examples"
TARGET_CLASS_ID = "10"
MAX_EXAMPLES = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

label_files = glob.glob(os.path.join(LABELS_DIR, "*.txt"))
found = 0

for label_path in label_files:
    if found >= MAX_EXAMPLES:
        break

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    # check whether this file contains class 10
    target_boxes = [l.split() for l in lines if l.split()[0] == TARGET_CLASS_ID]
    if not target_boxes:
        continue

    # find the matching image (same name, extension may differ)
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    image_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        continue

    img = cv2.imread(image_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # draw ALL boxes in the image (for context), with class 10 highlighted in thick red
    for parts in [l.split() for l in lines]:
        cls_id, xc, yc, bw, bh = parts[:5]
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        if cls_id == TARGET_CLASS_ID:
            color = (0, 0, 255)  # red = unknown class
            thickness = 3
            label = "CLASS 10 (unknown)"
        else:
            color = (0, 255, 0)  # green = other classes, for context only
            thickness = 1
            label = f"id={cls_id}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = os.path.join(OUTPUT_DIR, f"example_{found}_{base_name}.jpg")
    cv2.imwrite(out_path, img)
    found += 1
    print(f"Saved: {out_path}")

print(f"\nDone! {found} examples saved in the '{OUTPUT_DIR}/' folder.")
print("Open them and look at what's framed in RED (class 10).")