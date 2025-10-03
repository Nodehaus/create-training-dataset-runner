import json
import logging
import math
import os
import time
import traceback

import boto3
import requests
from dotenv import load_dotenv

from training_dataset.annotation import (
    generate_annotations,
    generate_annotations_with_text,
)
from training_dataset.runpod_client import RunpodClient

# Configure logging for all modules
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://s3.peterbouda.eu:3900")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "garage")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")
APP_S3_BUCKET = os.getenv("APP_S3_BUCKET", "")

AI_PLATFORM_API_BASE_URL = os.getenv("AI_PLATFORM_API_BASE_URL", "")
AI_PLATFORM_API_KEY = os.getenv("AI_PLATFORM_API_KEY", "")

DATASTES_PATH = "datasets/"
JOBS_PATH = "jobs/datasets/"
JOBS_DONE_PATH = "jobs_done/datasets/"
JOBS_FAILED_PATH = "jobs_failed/datasets/"


def update_training_dataset_status_api(training_dataset_id: str, status: str) -> bool:
    """Update training dataset status via API.

    Args:
        training_dataset_id: The ID of the training dataset
        status: The status to set (e.g., "RUNNING", "COMPLETED", "FAILED")

    Returns:
        True if successful, False otherwise
    """
    url = f"{AI_PLATFORM_API_BASE_URL}/api/external/training-datasets/{training_dataset_id}/update-status"
    headers = {
        "X-API-Key": AI_PLATFORM_API_KEY,
        "Content-Type": "application/json",
    }
    data = {"status": status}

    try:
        response = requests.put(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        logger.info(
            f"Updated training dataset {training_dataset_id} status to {status}"
        )
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to update training dataset status: {e}")
        logger.error(f"Response body: {response.text}")
        return False


def get_files_in_path(s3_client, path: str) -> list[str]:
    """Retrieve all files from the JOBS_PATH on S3.

    Args:
        s3_client: Configured boto3 S3 client
        path: The path to get the files for

    Returns:
        List of file keys under path
    """
    response = s3_client.list_objects_v2(Bucket=APP_S3_BUCKET, Prefix=path)

    if "Contents" not in response:
        return []

    # Extract file keys and remove the prefix
    files = []
    for obj in response["Contents"]:
        key = obj["Key"]
        # Skip directory markers (keys ending with '/')
        if not key.endswith("/"):
            files.append(key)

    return files


def create_annotations_with_corpus(
    s3_client,
    runpod_client,
    job_data: dict,
    annotations_s3_prefix: str,
    generate_examples_number: int,
    language_iso: str,
) -> None:
    """Create annotations from corpus documents.

    Args:
        s3_client: Configured boto3 S3 client
        runpod_client: Configured RunPod client
        job_data: Job configuration data
        annotations_s3_prefix: S3 prefix for annotation files
        generate_examples_number: Number of examples to generate
        language_iso: Language ISO code
    """
    corpus_s3_path = job_data["corpus_s3_path"].strip("/") + "/" + language_iso

    files_to_process = []
    if job_data["corpus_files_subset"]:
        files_to_process = [
            f"{corpus_s3_path}/{filename}"
            for filename in job_data["corpus_files_subset"]
        ]
    else:
        files_to_process = get_files_in_path(s3_client, corpus_s3_path)

    examples_per_document = max(
        1,
        math.ceil(generate_examples_number / len(files_to_process)),
    )
    examples_created = 0

    logger.info(
        f"Will process {len(files_to_process)} files, {examples_per_document} examples per document"
    )

    for document_filepath in files_to_process:
        logger.info(f"Processing file {document_filepath}...")

        document_filename = document_filepath.split("/")[-1]
        annotations_filepath = f"{annotations_s3_prefix}{document_filename}"

        try:
            s3_client.head_object(Bucket=APP_S3_BUCKET, Key=annotations_filepath)

            # Download file, count annotations and add to examples_created
            response = s3_client.get_object(
                Bucket=APP_S3_BUCKET, Key=annotations_filepath
            )
            annotation_data = json.loads(response["Body"].read().decode("utf-8"))
            existing_annotations_count = annotation_data.get("count_annotations")
            examples_created += existing_annotations_count
            logger.info(
                f"Added {existing_annotations_count} existing annotations to count"
            )
            continue
        except s3_client.exceptions.ClientError:
            # File doesn't exist, proceed with generation
            pass

        response = s3_client.get_object(Bucket=APP_S3_BUCKET, Key=document_filepath)
        doc_data = json.loads(response["Body"].read().decode("utf-8"))
        content = doc_data.get("content")

        start_time = time.time()
        annotations = generate_annotations_with_text(
            runpod_client,
            content,
            doc_data.get("id"),
            job_data["generate_prompt"],
            job_data["json_object_fields"],
            examples_per_document,
            language_iso=language_iso,
        )
        end_time = time.time()
        inference_time = end_time - start_time
        count_annotations = len(annotations)
        avg_inference_time_per_annotation = inference_time / count_annotations

        data = {
            "generate_model": job_data["generate_model"],
            "generate_model_runner": job_data["generate_model_runner"],
            "gpu_info": {},
            "input_field": job_data.get("input_field"),
            "output_field": job_data.get("output_field"),
            "count_annotations": len(annotations),
            "total_generation_time": round(inference_time, 3),
            "avg_generation_time_per_annotation": round(
                avg_inference_time_per_annotation, 3
            ),
            "training_dataset_id": job_data["training_dataset_id"],
            "user_id": job_data["user_id"],
            "annotations": annotations,
        }

        # Write annotations to S3
        s3_client.put_object(
            Bucket=APP_S3_BUCKET,
            Key=annotations_filepath,
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Annotations saved to S3 at {annotations_filepath}")

        examples_created += len(annotations)
        if examples_created >= job_data["generate_examples_number"]:
            logger.info(f"Loop exit as we created {examples_created} examples")
            break


def create_annotations(
    s3_client,
    runpod_client,
    job_data: dict,
    annotations_s3_prefix: str,
    generate_examples_number: int,
    language_iso: str,
) -> None:
    """Create annotations without corpus documents.

    Args:
        s3_client: Configured boto3 S3 client
        runpod_client: Configured RunPod client
        job_data: Job configuration data
        annotations_s3_prefix: S3 prefix for annotation files
        generate_examples_number: Number of examples to generate
        language_iso: Language ISO code
    """
    start_time = time.time()
    annotations = generate_annotations(
        runpod_client,
        job_data["generate_prompt"],
        job_data["json_object_fields"],
        job_data["expected_output_size_chars"],
        generate_examples_number,
        language_iso=language_iso,
    )
    end_time = time.time()
    inference_time = end_time - start_time
    count_annotations = len(annotations)
    avg_inference_time_per_annotation = inference_time / count_annotations

    data = {
        "generate_model": job_data["generate_model"],
        "generate_model_runner": job_data["generate_model_runner"],
        "gpu_info": {},
        "input_field": job_data.get("input_field"),
        "output_field": job_data.get("output_field"),
        "count_annotations": len(annotations),
        "total_generation_time": round(inference_time, 3),
        "avg_generation_time_per_annotation": round(
            avg_inference_time_per_annotation, 3
        ),
        "training_dataset_id": job_data["training_dataset_id"],
        "user_id": job_data["user_id"],
        "annotations": annotations,
    }

    annotations_filepath = f"{annotations_s3_prefix}dataset.json"
    # Write annotations to S3
    s3_client.put_object(
        Bucket=APP_S3_BUCKET,
        Key=annotations_filepath,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Annotations saved to S3 at {annotations_filepath}")


def process_jobs():
    """Generate annotations for documents stored in S3."""
    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        region_name=AWS_DEFAULT_REGION,
    )

    jobs = get_files_in_path(s3_client, JOBS_PATH)
    logger.info(f"Jobs to process: {jobs}")
    for current_job in jobs:
        logger.info(f"Processing job {current_job}")
        response = s3_client.get_object(Bucket=APP_S3_BUCKET, Key=current_job)
        job_data = json.loads(response["Body"].read().decode("utf-8"))
        logger.info(f"Jobs data: {job_data}")

        training_dataset_id = job_data.get("training_dataset_id")

        update_training_dataset_status_api(training_dataset_id, "RUNNING")

        language_iso = job_data.get("language_iso")

        # TODO: Improve error handling here, make it more specific when we know which
        # errors occur.
        try:
            runpod_client = RunpodClient(
                pod_id=RUNPOD_POD_ID,
                api_key=RUNPOD_API_KEY,
                model=job_data["generate_model"],
            )

            annotations_s3_prefix = f"{DATASTES_PATH}{job_data['training_dataset_id']}/"
            generate_examples_number = min(1000, job_data["generate_examples_number"])

            if job_data["corpus_s3_path"] != "":
                create_annotations_with_corpus(
                    s3_client,
                    runpod_client,
                    job_data,
                    annotations_s3_prefix,
                    generate_examples_number,
                    language_iso,
                )
            else:
                create_annotations(
                    s3_client,
                    runpod_client,
                    job_data,
                    annotations_s3_prefix,
                    generate_examples_number,
                    language_iso,
                )

            # Move finished job file to `jobs_done/`
            job_filename = current_job.split("/")[-1]
            jobs_done_key = f"{JOBS_DONE_PATH}{job_filename}"

            # Copy the job file to jobs_done/
            s3_client.copy_object(
                Bucket=APP_S3_BUCKET,
                CopySource={"Bucket": APP_S3_BUCKET, "Key": current_job},
                Key=jobs_done_key,
            )

            # Delete the original job file
            s3_client.delete_object(Bucket=APP_S3_BUCKET, Key=current_job)
            logger.info(
                f"Moved completed job file from {current_job} to {jobs_done_key}"
            )

            update_training_dataset_status_api(training_dataset_id, "DONE")

        except Exception as e:
            logger.error("Job failed!!!!!!!!!!!!!!!!!!!")

            # Move failed job file to `jobs_failed/`
            job_filename = current_job.split("/")[-1]
            jobs_failed_key = f"{JOBS_FAILED_PATH}{job_filename}"

            # Copy the job file to jobs_failed/
            s3_client.copy_object(
                Bucket=APP_S3_BUCKET,
                CopySource={"Bucket": APP_S3_BUCKET, "Key": current_job},
                Key=jobs_failed_key,
            )

            # Delete the original job file
            s3_client.delete_object(Bucket=APP_S3_BUCKET, Key=current_job)
            logger.info(
                f"Moved completed job file from {current_job} to {jobs_failed_key}"
            )

            update_training_dataset_status_api(training_dataset_id, "FAILED")
            raise e


def main():
    while True:
        try:
            process_jobs()
        except Exception:
            logger.error(traceback.format_exc())
            # For now we just continue to loop
            pass
        logger.info("Waiting to process next job...")
        time.sleep(60)


if __name__ == "__main__":
    main()
