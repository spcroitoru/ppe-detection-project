"""
Testeaza modelul antrenat (smoke test, 3 epoci) live, pe camera web.

ATENTIE: modelul e antrenat doar 3 epoci - e normal sa detecteze putin,
gresit, sau deloc. Scopul e sa confirmam ca partea de INFERENTA LIVE
functioneaza tehnic (cadru -> model -> afisare cutii), nu sa avem
deja un model bun.

Ruleaza din radacina proiectului:
    python webcam_test.py

Apasa 'q' in fereastra video ca sa inchizi.
"""
from ultralytics import YOLO

# folosim modelul salvat de smoke test (cel mai bun dupa cele 3 epoci)
MODEL_PATH = "C:/Users/croit/Documents/Proiect/ppe-detection-project/runs/detect/runs_test/smoke_test/weights/best.pt"

def main():
    model = YOLO(MODEL_PATH)

    # source=0 = prima camera web disponibila (built-in sau USB)
    # show=True = deschide o fereastra live cu cutiile desenate automat
    # conf=0.25 = arata doar detectiile cu incredere peste 25%
    # (scazut intentionat, ca modelul e slab antrenat inca)
    model.predict(
        source=0,
        show=True,
        conf=0.25,
        imgsz=416,
    )

if __name__ == "__main__":
    main()
