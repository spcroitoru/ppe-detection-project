"""
Testeaza modelul antrenat (smoke test, 3 epoci) live, pe camera web.
Cutiile sunt colorate manual: VERDE = echipament prezent/corect,
ROSU = echipament lipsa/incalcare (clase care incep cu "NO_"/"No_", sau "Slippers").

ATENTIE: modelul e antrenat doar 3 epoci - e normal sa detecteze putin,
gresit, sau deloc. Scopul e sa confirmam ca partea de INFERENTA LIVE
functioneaza tehnic (cadru -> model -> afisare cutii), nu sa avem
deja un model bun.

Ruleaza din radacina proiectului:
    python webcam_test.py

Apasa 'q' in fereastra video ca sa inchizi.
"""
import cv2
from ultralytics import YOLO

MODEL_PATH = "C:/Users/croit/Documents/Proiect/ppe-detection-project/runs/detect/runs_test/smoke_test/weights/best.pt"

# clase care semnaleaza lipsa echipamentului / incalcare -> ROSU
# restul claselor (Helmet, Safety_Vest, etc.) -> VERDE
CLASE_ROSU = {"NO_helmet", "NO_Vest", "NO_goggles", "No_SafetyShoes", "NO_Gloves", "Slippers"}

COLOR_VERDE = (0, 200, 0)   # BGR (OpenCV foloseste BGR, nu RGB)
COLOR_ROSU = (0, 0, 255)

def main():
    model = YOLO(MODEL_PATH)

    # stream=True = proceseaza continuu cadre de la camera, fara sa desenam automat (show=False)
    results_generator = model.predict(
        source=0,
        stream=True,
        conf=0.25,
        imgsz=416,
        show=False,
    )

    for result in results_generator:
        frame = result.orig_img.copy()

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = COLOR_ROSU if cls_name in CLASE_ROSU else COLOR_VERDE

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("PPE Detection - apasa 'q' pentru iesire", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()