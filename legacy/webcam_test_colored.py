"""
Tests the trained model (smoke test, 3 epochs) live, on the webcam.
Boxes are colored manually: GREEN = PPE present/correct,
RED = PPE missing/violation (classes starting with "NO_"/"No_", or "Slippers").

NOTE: the model is trained for only 3 epochs - it's expected to detect little,
incorrectly, or not at all. The goal is to confirm that the LIVE INFERENCE
part works technically (frame -> model -> box display), not to already
have a good model.

Run from the project root:
    python webcam_test.py

Press 'q' in the video window to quit.
"""
import cv2
from ultralytics import YOLO

MODEL_PATH = "C:/Users/croit/Documents/Proiect/ppe-detection-project/runs/detect/runs_test/smoke_test/weights/best.pt"

# classes that signal missing/violated PPE -> RED
# all other classes (Helmet, Safety_Vest, etc.) -> GREEN
RED_CLASSES = {"NO_helmet", "NO_Vest", "NO_goggles", "No_SafetyShoes", "NO_Gloves", "Slippers"}

COLOR_GREEN = (0, 200, 0)   # BGR (OpenCV uses BGR, not RGB)
COLOR_RED = (0, 0, 255)

def main():
    model = YOLO(MODEL_PATH)

    # stream=True = continuously process frames from the camera, without auto-drawing (show=False)
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

            color = COLOR_RED if cls_name in RED_CLASSES else COLOR_GREEN

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("PPE Detection - press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()