"""
Upload and trigger Kubeflow Pipeline run on local Kind KFP cluster using parameters from params.yaml.
"""

import sys
import yaml
import kfp

KFP_HOST = "http://127.0.0.1:8080"
PIPELINE_PATH = "pipelines/house_price_pipeline.yaml"
EXPERIMENT_NAME = "house-price-experiment"

try:
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    num_samples = config.get("data_generation", {}).get("num_samples", 5000)
    random_state = config.get("base", {}).get("random_state", 42)
    test_size = config.get("preprocessing", {}).get("test_size", 0.2)
    n_estimators = config.get("training", {}).get("random_forest", {}).get("n_estimators", 100)

    client = kfp.Client(host=KFP_HOST)
    print(f"[KFP] Connecting to {KFP_HOST}...")

    # Upload pipeline if not already present
    pipeline_name = "House-Price-Prediction-Pipeline"
    pipeline_id = None
    try:
        pipeline = client.upload_pipeline(
            pipeline_package_path=PIPELINE_PATH,
            pipeline_name=pipeline_name,
            description="End-to-End MLOps Pipeline: Generate, Validate, Preprocess, Train & Evaluate",
        )
        pipeline_id = pipeline.pipeline_id
        print(f"[KFP] Uploaded Pipeline ID: {pipeline_id}")
    except Exception:
        # If already uploaded, find it from the list
        pipes = client.list_pipelines().pipelines or []
        for p in pipes:
            if p.display_name == pipeline_name or p.name == pipeline_name:
                pipeline_id = p.pipeline_id
                break

    # Trigger a run with current params.yaml arguments
    run_name = f"run-samples-{num_samples}"
    run = client.create_run_from_pipeline_package(
        pipeline_file=PIPELINE_PATH,
        experiment_name=EXPERIMENT_NAME,
        run_name=run_name,
        arguments={
            "num_samples": num_samples,
            "test_size": test_size,
            "n_estimators": n_estimators,
            "random_state": random_state,
        },
    )
    print(f"[KFP] Triggered Run ID: {run.run_id} with num_samples={num_samples}")
    print(f"[KFP] Run Details URL: {KFP_HOST}/#/runs/details/{run.run_id}")

except Exception as e:
    print(f"[KFP] Error: {e}", file=sys.stderr)
    sys.exit(1)

