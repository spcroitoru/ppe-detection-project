"""
Manually logs the final results of the imgsz=640+SGD experiment into MLflow,
since the resumed segment (epochs 42-50) failed to auto-log due to a missing
MLFLOW_TRACKING_URI environment variable in the standalone resume script.
"""
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ppe-detection")

with mlflow.start_run(run_name="pipeline_test_imgsz640_sgd_manual_import"):
    mlflow.log_params({
        "model_name": "yolov8n.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 8,
        "optimizer": "SGD",
        "patience": 10,
        "lr0": 0.01,
        "lrf": 0.01,
    })

    mlflow.log_metrics({
        "mAP50": 0.708,
        "mAP50-95": 0.420,
        "precision": 0.771,
        "recall": 0.707,
        "mAP50_Helmet": 0.936,
        "mAP50_Safety_Vest": 0.893,
        "mAP50_Safety_goggles": 0.608,
        "mAP50_Safety_shoes": 0.761,
        "mAP50_NO_helmet": 0.877,
        "mAP50_NO_Vest": 0.981,
        "mAP50_NO_goggles": 0.372,
        "mAP50_No_SafetyShoes": 0.555,
        "mAP50_Person": 0.990,
        "mAP50_Slippers": 0.259,
        "mAP50_NO_Gloves": 0.555,
    })

    mlflow.log_artifact(
        "runs/detect/runs_test/exp_imgsz640_sgd_final/weights/best.pt",
        artifact_path="weights"
    )

    print("Manual import complete - check MLflow UI for the new run.")