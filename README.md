# PPE Detection - AI Final Project

Detection of personal protective equipment (PPE) on construction workers, using
YOLO (PyTorch), with live deployment on a webcam.

Dataset: [PPE Kit Detection - Kaggle](https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers)

## Project structure

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

## Step progress

- [x] Step 1: Repo + Python env
- [x] Step 2: Working dataset example + source code
- [x] Step 3: EDA
- [x] Step 4: MLflow integration
- [x] Step 5: Code refactoring
- [ ] Step 6: Train/val/test pipeline + early stopping + LR scheduler
- [ ] Step 7: Experimenting with different configurations
- [ ] Step 8: Tests + CI
- [ ] Step 9: Automated Docker build (GitHub Actions)
- [ ] Step 10: PPT presentation