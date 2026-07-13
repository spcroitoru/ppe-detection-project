"""
Appends the final results (epochs 42-50) directly to the ORIGINAL MLflow run
(rather than creating a separate manual-import run), so the experiment stays
as a single, continuous record.
"""
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

ORIGINAL_RUN_ID = "c120d8eaea414214ae30ab66758fe8c2"

with mlflow.start_run(run_id=ORIGINAL_RUN_ID):
    mlflow.log_metrics({
        "mAP50_final": 0.708,
        "mAP50-95_final": 0.420,
        "precision_final": 0.771,
        "recall_final": 0.707,
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
    }, step=50)

    mlflow.log_artifact(
        "runs/detect/runs_test/exp_imgsz640_sgd_final/weights/best.pt",
        artifact_path="weights"
    )

    print("Successfully appended final results to the original run.")