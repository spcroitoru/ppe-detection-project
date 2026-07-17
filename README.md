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

**Experiment 2: yolov8s architecture** (run: `exp_yolov8s_imgsz640_sgd`, ~7h,
  50 epochs complete, no early stopping triggered)

  Building on the best configuration found so far (imgsz=640, SGD optimizer),
  tested whether a larger model architecture (yolov8s, ~11.1M parameters vs
  yolov8n's ~3.0M) further improves detection accuracy. This isolates a single
  new variable (architecture) against the winning imgsz=640+SGD configuration.

  **Final results (50 epochs complete):**

  | Metric | imgsz=640+SGD (yolov8n) | imgsz=640+SGD (yolov8s) |
  |---|---|---|
  | mAP50 | 0.708 | **0.747** |
  | mAP50-95 | 0.420 | **0.473** |

  Per-class results (yolov8s):

  | Class | mAP50 | mAP50-95 |
  |---|---|---|
  | Helmet | 0.961 | 0.670 |
  | Safety_Vest | 0.910 | 0.585 |
  | Safety_goggles | 0.664 | 0.389 |
  | Safety_shoes | 0.828 | 0.447 |
  | NO_helmet | 0.906 | 0.545 |
  | NO_Vest | 0.983 | 0.804 |
  | NO_goggles | 0.435 | 0.184 |
  | No_SafetyShoes | 0.638 | 0.326 |
  | Person | 0.993 | 0.871 |
  | Slippers | 0.239 | 0.162 |
  | NO_Gloves | 0.660 | 0.219 |

  **Conclusion:** the larger architecture provided a further, consistent
  improvement across nearly all classes, confirming that additional model
  capacity helps this task. `Slippers` remains the weakest class throughout
  all three experiments (57 training images only), reinforcing the EDA
  finding that this is primarily a data-quantity limitation, not a training
  configuration issue.

  **Summary across all Step 6/7 experiments:**

  | # | Configuration | mAP50 | mAP50-95 |
  |---|---|---|---|
  | 1 | Step 6 baseline: yolov8n, imgsz=416, AdamW | 0.574 | 0.322 |
  | 2 | yolov8n, imgsz=640, SGD | 0.708 | 0.420 |
  | 3 | **yolov8s, imgsz=640, SGD** | **0.747** | **0.473** |
  Best model: `exp_yolov8s_imgsz640_sgd`

  ## Step 8 notes - Tests + CI

- Added `tests/test_data_utils.py` (9 tests) covering preprocessing functions:
  `list_images`, `check_data_quality`, `clean_labels`, `convert_jfif_to_jpg`,
  `resplit_train_val`. Uses temporary directories with synthetic data for
  fast, isolated tests.
- Added `tests/test_inference.py` (2 tests) covering postprocessing:
  `draw_detections`, using mock objects to simulate Ultralytics' detection
  results without requiring a real model or camera.
- Added `.github/workflows/tests.yml`: runs `pytest tests/` automatically on
  every push/PR to `main`, via GitHub Actions.
- **CI troubleshooting:** initial CI runs failed because `src/inference.py`
  imported `ultralytics` (and therefore `torch`) at module level. Since the
  CI environment intentionally doesn't install these heavy dependencies (to
  keep test runs fast), importing the module for testing `draw_detections`
  failed with `ModuleNotFoundError`. Fixed by moving the `ultralytics`
  import inside `run_webcam_inference()`, where it's actually needed -
  `draw_detections()` itself has no dependency on it.

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
  pochs 42-50 in a new one). This was then corrected by re-opening the
  *original* run via its `run_id` (`mlflow.start_run(run_id=...)`) and logging
  the final metrics and model artifact directly into it, unifying the full
  1-50 epoch experiment into a single MLflow run. The separate, now-redundant
  run was deleted for a clean experiment history.

  *(Note: MLflow's displayed "Duration" for this run reflects wall-clock time
  across multiple sessions - including the power interruption, transport
  pause, and manual backfill steps - not pure GPU compute time.)*

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
7.  When combining multiple experimental variables (e.g. resolution + optimizer,
   or architecture + resolution + optimizer) in a single run due to time
   constraints, document this explicitly as a methodological limitation -
   results can't be attributed to a single cause, but the combined
   configuration is still valid to report.
8. Bigger isn't always slower in the way you'd expect: yolov8s took ~2x the
   time per epoch compared to yolov8n at the same settings, but delivered
   proportionally larger accuracy gains.
9. Consistently, the weakest-performing classes across all three experiments
   (`Slippers`, `Safety_goggles`) were the ones with the fewest training
   examples (57 and 18 images respectively) - reinforcing that, past a
   certain point, more epochs/better optimizers/bigger models can't fully
   compensate for insufficient data on a specific class.
10. Unit tests for a small, well-defined set of preprocessing/postprocessing
   functions (not the entire codebase) is enough to satisfy a "testing"
   requirement - exhaustive coverage isn't the goal, demonstrating the
   practice is.
11. CI environments are intentionally minimal - installing heavy dependencies
   (like `torch`/`ultralytics`) just to test a small utility function is
   wasteful and slows down every push. Structuring code so that heavy
   imports are deferred to inside the functions that actually need them
   (rather than at module level) keeps modules testable in lightweight
   environments.
12. A module-level import failing during test collection (e.g.
   `ModuleNotFoundError`) aborts the *entire* test file's collection, not
   just the tests that would have used that import - one bad import at the
   top of a file can silently hide otherwise-passing tests.
13. Always verify a file's actual saved content (`type filename`) before
   assuming an edit was applied correctly, especially after copy-pasting
   similar code between multiple files (e.g. accidentally pasting
   `inference.py`'s content into `test_inference.py`) - this caused two
   separate rounds of CI failures before being caught.
14. GitHub Actions occasionally shows deprecation warnings (e.g. Node.js
   version) for third-party actions unrelated to the project's own code -
   these don't block the pipeline and can be addressed later by bumping
   action versions (e.g. `actions/checkout@v4` -> `@v5`).

## Step progress

- [x] Step 1: Repo + Python env
- [x] Step 2: Working dataset example + source code
- [x] Step 3: EDA
- [x] Step 4: MLflow integration
- [x] Step 5: Code refactoring
- [x] Step 6: Train/val/test pipeline + early stopping + LR scheduler
- [x] Step 7: Experimenting with different configurations
- [x] Step 8: Tests + CI
- [ ] Step 9: Automated Docker build (GitHub Actions)
- [ ] Step 10: PPT presentation