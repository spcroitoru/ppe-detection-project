from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("runs/detect/runs_test/pipeline_test/weights/last.pt")
    model.train(resume=True)