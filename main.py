import json
import os
import time

import boto3
from dotenv import load_dotenv

from training_dataset.annotation import (
    QuestionAnswerGenerator,
)
from training_dataset.runpod_client import RunpodClient

# Load environment variables from .env file
load_dotenv()

MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"

AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://s3.peterbouda.eu:3900")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "garage")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")

DOCUMENTS_S3_BUCKET = os.getenv("DOCUMENTS_S3_BUCKET", "nodehaus")
DOCUMENTS_PATH = "/documents/eurlex"
DOCUMENTS_CORPUS_ID = "eurlex"
DOCUMENTS_LANGUAGE = "eng"
DOCUMENTS = [
    "02016R0679-20160504_eng.json",
    "02002L0058-20091219_eng.json",
    "02016L0680-20160504_eng.json",
    "02023R1115-20241226_eng.json",
    "02010L0075-20240804_eng.json",
    "02006R1907-20250623_eng.json",
    "02008L0098-20180705_eng.json",
    "02014L0065-20250117_eng.json",
    "02013R0575-20250629_eng.json",
    "02013L0036-20250117_eng.json",
    "02015L2366-20250117_eng.json",
    "02016R1011-20220101_eng.json",
]

DATASTES_PATH = "/datasets"
JOBS_PATH = "/jobs"

def main() -> None:
    """Generate annotations for documents stored in S3."""
    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        region_name=AWS_DEFAULT_REGION,
    )

    runpod_client = RunpodClient(pod_id=RUNPOD_POD_ID, api_key=RUNPOD_API_KEY)

    # S3 paths
    documents_s3_prefix = (
        f"{DOCUMENTS_PATH}/{DOCUMENTS_LANGUAGE}/"
    )
    annotations_s3_prefix = f"{ANNOTATIONS_BASE_PATH.lstrip('/')}/{DOCUMENTS_CORPUS_ID}/{DOCUMENTS_LANGUAGE}/"

    for document_filename in DOCUMENTS:
        annotations_id = "qa"
        annotations_class = QuestionAnswerGenerator
        print(f"Generating {annotations_id} for {document_filename}")

        # Check if annotation already exists in S3
        annotations_s3_key = (
            f"{annotations_s3_prefix}{annotations_id}/{document_filename}"
        )
        try:
            s3_client.head_object(
                Bucket=DOCUMENTS_S3_BUCKET, Key=annotations_s3_key
            )
            print("  Annotation file exists in S3, skipping...")
            continue
        except s3_client.exceptions.ClientError:
            # File doesn't exist, proceed with generation
            pass

        # Read document from S3
        document_s3_key = f"{documents_s3_prefix}{document_filename}"
        response = s3_client.get_object(
            Bucket=DOCUMENTS_S3_BUCKET, Key=document_s3_key
        )
        doc_data = json.loads(response["Body"].read().decode("utf-8"))

        content = doc_data.get("content")

        start_time = time.time()
        annotations = annotations_class.generate_annotations(
            runpod_client, content, doc_data.get("id")
        )
        end_time = time.time()
        inference_time = end_time - start_time
        count_annotations = len(annotations)
        avg_inference_time_per_annotation = inference_time / count_annotations

        data = {
            "model_name": MODEL,
            "model_runner": "ollama",
            "gpu_info": {},
            "input_field": annotations_class.input_field,
            "output_field": annotations_class.output_field,
            "count_annotations": len(annotations),
            "total_generation_time": round(inference_time, 3),
            "avg_generation_time_per_annotation": round(
                avg_inference_time_per_annotation, 3
            ),
            "annotations_id": annotations_id,
            "annotations": annotations,
        }

        # Write annotations to S3
        s3_client.put_object(
            Bucket=DOCUMENTS_S3_BUCKET,
            Key=annotations_s3_key,
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"  Annotations saved to S3: {annotations_s3_key}")


if __name__ == "__main__":
    main()
