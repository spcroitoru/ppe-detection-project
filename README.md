# PPE Detection - Proiect Final AI

Detectare echipament de protecție (PPE) pe muncitori din construcții, folosind
YOLO (PyTorch), cu deployment live pe cameră web.

Dataset: [PPE Kit Detection - Kaggle](https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers)

## Structura proiectului (se va completa progresiv, pe parcursul pașilor)

```
ppe-detection-project/
├── data/                   # dataset (NU se comite in git - vezi .gitignore)
│   ├── train/
│   ├── valid/
│   └── test/
├── notebooks/
│   └── 01_eda.ipynb        # analiza exploratorie a datelor (Pas 3)
├── src/
│   ├── config.py           # hiperparametri, path-uri (Pas 5)
│   ├── data_utils.py       # incarcare/split date, preprocesare (Pas 5-6)
│   ├── train.py            # bucla de antrenare + MLflow logging (Pas 4-5)
│   ├── evaluate.py         # evaluare pe test set
│   └── inference_webcam.py # detectie live pe camera web
├── tests/
│   └── test_data_utils.py  # teste unitare (Pas 8)
├── .github/
│   └── workflows/
│       ├── tests.yml       # CI: ruleaza testele la fiecare commit (Pas 8)
│       └── docker-build.yml# CI: build imagine docker (Pas 9)
├── mlruns/                 # experimente MLflow (generat automat)
├── Dockerfile              # (Pas 9)
├── requirements.txt
├── .gitignore
└── README.md
```

## Pas 1 - Setup mediu

```bash
git init
python -m venv venv

# activare mediu virtual
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Status pași

- [x] Pas 1: Repo + env Python
- [x] Pas 2: Exemplu dataset + cod sursa functional
- [x] Pas 3: EDA
- [x] Pas 4: Integrare MLflow
- [ ] Pas 5: Refactorizare cod
- [ ] Pas 6: Pipeline train/val/test + early stopping + LR scheduler
- [ ] Pas 7: Experimentare configuratii
- [ ] Pas 8: Teste + CI
- [ ] Pas 9: Docker build automat (GitHub Actions)
- [ ] Pas 10: Prezentare PPT
