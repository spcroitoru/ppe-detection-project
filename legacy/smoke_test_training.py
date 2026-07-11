"""
Test end-to-end (Pas 2): confirmam ca datele se incarca si antrenarea
ruleaza cap-coada pe GPU, inainte sa trecem la EDA complet (Pas 3) si
la integrarea MLflow (Pas 4).

Deliberat MINIMAL:
- model cel mai mic din familia YOLOv8 (yolov8n = "nano") - potrivit pentru
  o placa cu doar 4GB VRAM (GTX 1050 Ti)
- doar 3 epoci - suficient sa vedem ca antrenarea scade loss-ul si ruleaza
  pe GPU, fara sa asteptam ore intregi
- imgsz mic (416 in loc de implicitul 640) - reduce memoria folosita

Ruleaza din radacina proiectului:
    python test_training.py
"""
import os

# Conectare la serverul MLflow local
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "ppe-detection"

# Ultralytics detectează automat MLflow și face autolog dacă e instalat

from ultralytics import YOLO
import torch


def main():
    print("CUDA disponibil:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (ruleaza pe CPU)")

    # yolov8n.pt = model pre-antrenat pe COCO, il descarca automat prima data (~6MB)
    # pornim de la el (transfer learning) in loc de la zero - mult mai rapid si eficient
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data.yaml",
        epochs=3,
        imgsz=416,
        batch=8,          # mic, ca sa incapa sigur in 4GB VRAM
        device=0,         # 0 = prima placa video (GPU); foloseste "cpu" daca vrei fallback
        project="runs_test",
        name="smoke_test",
        exist_ok=True,
    )

    print("\n--- Antrenare de test finalizata ---")
    print("Rezultate salvate in: runs_test/smoke_test/")
    print("Verifica graficele (results.png) si greutatile (weights/best.pt) generate acolo.")

if __name__ == "__main__":
    main()
