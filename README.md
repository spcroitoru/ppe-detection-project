## Step 1 - Environment setup
 
```bash
git init
python -m venv venv

# activate virtual environment
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch is installed separately, depending on hardware:

```bash
# NVIDIA GPU with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify GPU availability:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
All code, comments, and docstrings translated to English.

## Step 2 - Dataset & baseline example

Dataset: PPE Kit Detection (Kaggle), YOLO-format annotations (bounding boxes for
helmets, vests, goggles, shoes, gloves, and their "missing" counterparts).

Chosen stack: **PyTorch + Ultralytics YOLO** (object detection, not just
classification - needed for bounding boxes). Runs on both CPU and GPU, with
real-time webcam support out of the box.

A minimal smoke test (`legacy/smoke_test_training.py`) confirmed the full
pipeline works end-to-end: load data -> train a few epochs on GPU -> save
weights, before moving to full EDA and MLflow integration.

## Step 3 - Exploratory Data Analysis (EDA)

Notebook: `notebooks/01_eda.ipynb`. Covers:
- General inventory (image counts per split, resolutions)
- Class distribution (11 classes, bar chart)
- Missing values / structural errors check (orphan labels, images without
  labels, empty labels, invalid bounding boxes)
- Bounding box characteristics (aspect ratio, size, objects per image)
- Correlations (class co-occurrence, box size per class)
- Visual inspection (annotated image mosaic)

**Findings & fixes during EDA:**
- The original `data.yaml` from Kaggle had `nc: 9` but 10 class names listed
  (inconsistent). Label analysis revealed 11 distinct class ids (0-10). Class
  10 was missing from the names list entirely - visually identified as "hands
  without gloves" and named `NO_Gloves`, following the existing naming
  convention (`NO_helmet`, `NO_Vest`, `NO_goggles`).
- `list_images()` initially missed `.webp` and `.jfif` files, causing false
  "orphan label" counts (169/51/22 across splits). Fixed by including all
  actual image extensions present in the dataset.
- ~63 empty label files (images with no annotated objects) were kept as
  negative samples rather than removed - a normal and useful pattern for
  object detection training.

## Step 4 - MLflow integration

Local MLflow tracking server (`mlflow ui`, SQLite backend). Ultralytics YOLO
auto-detects MLflow via environment variables (`MLFLOW_TRACKING_URI`,
`MLFLOW_EXPERIMENT_NAME`) and automatically logs parameters (epochs, batch
size, optimizer, learning rate), metrics (loss, mAP, precision, recall per
class), and the best model checkpoint - no manual logging code needed for
the baseline integration.

**Known issue fixed:** MLflow 3.14.0 crashes on Python 3.14 with
`ImportError: cannot import name 'Traversable' from 'importlib.abc'`
(a confirmed upstream bug, not yet patched on PyPI at the time). Worked
around by patching the import in
`venv/Lib/site-packages/mlflow/assistant/skill_installer.py` with a
try/except fallback to the new `importlib.resources.abc` location.

## Step 5 - Code refactoring

Reorganized the initial flat scripts into a clean structure:
- `src/` - reusable logic (config, data utilities, training, inference)
- `scripts/` - thin entry points that call into `src/`
- `legacy/` - original pre-refactor scripts, kept for reference

Refactoring was validated by re-running the smoke test through the new structure and
comparing MLflow metrics against the original run (mAP50 within ~0.01 of
each other, confirming functional equivalence).

## Step 6 - Training pipeline improvements

**Data fix:** discovered that `.jfif` images (94 files across train+val) were
silently skipped by Ultralytics during training scans, because `.jfif` isn't
in Ultralytics' recognized `IMG_FORMATS` list (`avif, bmp, dng, heic, heif,
jp2, jpeg, jpg, mpo, png, tif, tiff, webp`). Fixed by renaming all `.jfif`
files to `.jpg` (binary-identical format, safe rename) via
`convert_jfif_to_jpg()` in `src/data_utils.py`.

**Train/val re-split:** the original dataset was split 70/20/10
(train/val/test). Per project requirements, re-split train+val to 90/10
while keeping the test set completely untouched, via `resplit_train_val()`
in `src/data_utils.py`. This generates `train_split.txt` and `val_split.txt`
(list files consumed directly by `data.yaml`), without moving any files on
disk.

**Early stopping:** enabled via `patience=10` in `train_model()` - training
stops automatically if validation loss doesn't improve for 10 consecutive
epochs, preventing wasted compute and overfitting.

**Learning rate reduction:** enabled via `lr0` (initial) / `lrf` (final
factor) parameters - Ultralytics' built-in scheduler gradually reduces the
learning rate over the course of training.

## Step progress

- [x] Step 1: Repo + Python env
- [x] Step 2: Working dataset example + source code
- [x] Step 3: EDA
- [x] Step 4: MLflow integration
- [x] Step 5: Code refactoring
- [x] Step 6: Train/val/test pipeline + early stopping + LR scheduler
- [ ] Step 7: Experimenting with different configurations
- [ ] Step 8: Tests + CI
- [ ] Step 9: Automated Docker build (GitHub Actions)
- [ ] Step 10: PPT presentation