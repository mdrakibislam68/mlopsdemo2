"""
Upload trained model artifacts from local `models/` directory to MinIO under a `model/` folder.
"""

import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def upload_models_to_minio(
    models_dir: str = "models",
    bucket: str = "mlflow-artifacts",
    prefix: str = "model",
) -> list[str]:
    """Upload serialized model files to an S3-compatible MinIO bucket/prefix."""
    endpoint_url = os.getenv("MINIO_S3_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    # Ensure target bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        print(f"[MinIO] Creating bucket '{bucket}'...")
        s3.create_bucket(Bucket=bucket)

    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory '{models_dir}' does not exist.")

    uploaded: list[str] = []
    clean_prefix = prefix.strip("/")

    for file_path in sorted(models_path.iterdir()):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        object_key = f"{clean_prefix}/{file_path.name}" if clean_prefix else file_path.name
        print(f"[MinIO] Uploading '{file_path}' -> 's3://{bucket}/{object_key}'...")
        s3.upload_file(str(file_path), bucket, object_key)
        uploaded.append(object_key)

    print(f"[MinIO] Uploaded {len(uploaded)} model file(s) successfully.")
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Push local model artifacts to MinIO")
    parser.add_argument("--models-dir", default="models", help="Local models directory")
    parser.add_argument(
        "--bucket",
        default=os.getenv("MINIO_MLFLOW_BUCKET", "mlflow-artifacts"),
        help="Target MinIO bucket",
    )
    parser.add_argument("--prefix", default="model", help="Target folder prefix in bucket")
    args = parser.parse_args()

    upload_models_to_minio(
        models_dir=args.models_dir,
        bucket=args.bucket,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()

