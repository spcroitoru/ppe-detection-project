# Legacy

Scripts from the initial phase of the project (Step 2), before refactoring (Step 5).
Kept for historical reference, but functionally replaced by:

- `smoke_test_training.py` -> `scripts/run_training.py` + `src/train.py`
- `webcam_test.py` (simple variant, default Ultralytics colors) -> `scripts/run_webcam.py` + `src/inference.py`
- `webcam_test_colored.py` (manual green/red color variant) -> `scripts/run_webcam.py` + `src/inference.py`

No longer actively maintained and may not work if the project structure changes.