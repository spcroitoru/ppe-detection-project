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

## Step 7 notes - Experimentation

## Step 7 notes - Experimentation

**Experiment 1: imgsz=640 + SGD optimizer** (run: `pipeline_test`, ~7h total training time
across interruptions)

Testing whether higher input resolution (640 vs 416, Step 6 baseline) combined with an
explicit SGD optimizer (instead of Ultralytics' auto-selected AdamW) improves detection
accuracy, especially for small/underrepresented objects. Resolution and optimizer were
changed together in a single run rather than isolated separately, due to time constraints -
a known methodological limitation, documented here rather than tested for.


**Final results (50 epochs complete):**

| Metric | Step 6 baseline (imgsz=416, AdamW) | This experiment (imgsz=640, SGD) |
|---|---|---|
| mAP50 | 0.574 | **0.708** |
| mAP50-95 | 0.322 | **0.420** |

Per-class improvements were substantial across the board, including previously weak
classes:
- `Helmet`: 0.776 -> 0.936
- `NO_Vest`: 0.892 -> 0.981
- `Person`: 0.971 -> 0.990
- `NO_Gloves`: 0.137 -> 0.555 (largest relative improvement)
- `Safety_goggles`: 0.609 -> 0.608 (unchanged - still limited by very few examples, 18 images)
- `Slippers`: 0.206 -> 0.259 (modest improvement - still limited by few examples, 57 images)

**Conclusion:** the imgsz=640 + SGD combination clearly outperforms the Step 6 baseline.
Remaining weak classes (`Safety_goggles`, `Slippers`) appear to be limited primarily by
dataset size/imbalance rather than training configuration, consistent with the EDA findings.

**Troubleshooting - training interruptions and recovery:**
- The training machine lost power partway through the run, interrupting the process
  mid-training (around epoch 36-42, across two separate interruptions - one full power
  loss, one intentional pause for transport).
- **Root cause of a data-loss risk:** `scripts/run_training.py` had not been updated with
  a unique `name` parameter for this experiment - it reused `name="pipeline_test"` from
  the Step 6 baseline run. Restarting training with this script would have overwritten the
  Step 6 baseline's saved weights.
- **Recovery:** the Step 6 baseline model (imgsz=416, 50 epochs, mAP50=0.574) was
  recovered from MLflow's artifact store (`mlartifacts/<experiment_id>/<run_id>/artifacts/weights/best.pt`),
  which keeps a separate, run-specific copy independent of the `runs_test/` working
  directory - this is what made recovery possible.
- **Resuming training:** Ultralytics supports `model.train(resume=True)` when loading
  from a `last.pt` checkpoint, restoring the full training state (optimizer, learning
  rate schedule, early-stopping patience counter). On Windows, this requires wrapping
  the script in `if __name__ == "__main__":`, since Ultralytics' parallel data loading
  workers otherwise fail to spawn correctly.
- **Important discovery:** a checkpoint (`last.pt`) internally embeds its original run
  configuration (including `name`), which Ultralytics uses on resume regardless of the
  file's current folder location - renaming the containing folder does not change where
  resumed training writes its output. This caused some confusion when a folder was
  renamed mid-experiment; renaming should only be done after training is fully complete.
- **MLflow tracking gap:** a standalone `resume_training.py` script (not going through
  `src/train.py`) did not set the `MLFLOW_TRACKING_URI` environment variable, causing
  MLflow to fail silently for the final resumed segment (epochs 42-50): `WARNING MLflow:
  Failed to initialize... Not tracking this run`. The final metrics were still captured
  from the training console output and recorded here manually; the trained model
  weights themselves were unaffected (saved to disk independently of MLflow).
- **Performance note:** running on battery power (after an unexpected disconnect from
  mains power) caused significant GPU throttling - epoch time increased from ~8.5 min to
  ~29 min. Reconnecting to power restored normal speed. This did not affect training
  correctness, only wall-clock duration.
  **Manual MLflow backfill (iterative process):** the resumed training segment
(epochs 42-50) initially didn't appear in MLflow at all, due to the tracking
URI issue described above. The first fix attempt created a *separate* MLflow
run (`pipeline_test_imgsz640_sgd_manual_import`) with the final metrics and
model artifact manually logged - functional, but left the experiment split
across two disconnected runs (epochs 1-42 auto-logged in the original run,
epochs 42-50 in a new one). This was then corrected by re-opening the
*original* run via its `run_id` (`mlflow.start_run(run_id=...)`) and logging
the final metrics and model artifact directly into it, unifying the full
1-50 epoch experiment into a single MLflow run. The separate, now-redundant
run was deleted for a clean experiment history.

*(Note: MLflow's displayed "Duration" for this run reflects wall-clock time
across multiple sessions - including the power interruption, transport
pause, and manual backfill steps - not pure GPU compute time.)*

**Lessons learned:**
**Lessons learned:**
1. Always assign a unique, descriptive `name` per experiment *before* starting training.
2. Any standalone resume/training script must explicitly set MLflow environment
   variables, matching what `src/train.py` does - it's not inherited automatically.
3. Keep the laptop connected to mains power for any multi-hour training run.
4. MLflow's artifact store (`mlartifacts/`) is a reliable recovery point if the working
   directory (`runs_test/`) is accidentally overwritten.
5. When backfilling missing MLflow data after a tracking failure, prefer re-opening
   the *original* run via `mlflow.start_run(run_id=...)` rather than creating a new
   run - this keeps the full experiment history in one place instead of splitting it
   across multiple disconnected runs that then need manual cleanup.
6. MLflow metric names cannot contain parentheses or most special characters
   (e.g. `mAP50(B)` is invalid) - stick to alphanumerics, underscores, dashes,
   periods, spaces, and slashes.

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