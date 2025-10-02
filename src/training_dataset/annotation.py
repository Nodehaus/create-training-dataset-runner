import json
import logging
import math
import time
from typing import Any

from tqdm import tqdm

from .base_client import BaseClient
from .utils import split_text_into_chunks_with_offsets

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class JsonNotFoundError(Exception):
    """Raised when JSON response cannot be found or parsed from LLM output."""

    pass


def generate_annotations(
    llm_client: BaseClient,
    generate_prompt: str,
    examples_to_create: int,
    language_iso: str = "eng",
) -> list[dict[str, Any]]:
    retry_count = 0
    annotations = []

    while retry_count < MAX_RETRIES:
        prompt = _create_prompt(generate_prompt, examples_to_create, language_iso)
        start_time = time.time()
        response_text = llm_client.generate(prompt)
        end_time = time.time()
        inference_time = end_time - start_time

        try:
            annotations = _parse_response(response_text)
            break  # Success, exit retry loop
        except (JsonNotFoundError, json.decoder.JSONDecodeError):
            retry_count += 1
            if retry_count < MAX_RETRIES:
                logger.warning(
                    f"JSON parsing error, retrying "
                    f"({retry_count}/{MAX_RETRIES}), "
                    f"response was: {response_text}"
                )
            else:
                logger.error(
                    f"JSON parsing error after {MAX_RETRIES} retries, "
                    f"skipping chunk, response was: {response_text}"
                )

    for annotation in annotations:
        annotation["inference_time_seconds"] = round(inference_time, 3)

    return annotations


def generate_annotations_with_text(
    llm_client: BaseClient,
    text: str,
    document_id: str,
    generate_prompt: str,
    examples_to_create: int,
    use_chunking: bool = True,
    language_iso: str = "eng",
) -> list[dict[str, Any]]:
    """
    Generate annotations for a given text.

    Args:
        ollama_client: The Ollama client to use for generation
        text: The document text to analyze
        document_id: Identifier for the document
        use_chunking: Whether to split text into chunks (default: True)

    Returns:
        List of annotation dictionaries
    """
    if use_chunking:
        chunks_with_offsets = split_text_into_chunks_with_offsets(
            text, max_chunk_size=4000, overlap=100
        )
    else:
        chunks_with_offsets = [(text, 0, len(text))]

    all_annotations = []
    examples_per_chunk = max(
        1, math.ceil(examples_to_create / len(chunks_with_offsets))
    )
    logger.info(f"Creating {examples_per_chunk} examples per chunk")

    for chunk, start_offset, end_offset in tqdm(chunks_with_offsets):
        retry_count = 0
        annotations = []

        while retry_count < MAX_RETRIES:
            prompt = _create_prompt_with_text(
                chunk, generate_prompt, examples_per_chunk, language_iso
            )
            start_time = time.time()
            response_text = llm_client.generate(prompt)
            end_time = time.time()
            inference_time = end_time - start_time

            try:
                annotations = _parse_response(response_text)
                break  # Success, exit retry loop
            except (JsonNotFoundError, json.decoder.JSONDecodeError):
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    logger.warning(
                        f"JSON parsing error, retrying "
                        f"({retry_count}/{MAX_RETRIES}), "
                        f"response was: {response_text}"
                    )
                else:
                    logger.error(
                        f"JSON parsing error after {MAX_RETRIES} retries, "
                        f"skipping chunk, response was: {response_text}"
                    )

        for annotation in annotations:
            annotation["document_id"] = document_id
            annotation["within_start"] = start_offset
            annotation["within_end"] = end_offset
            annotation["inference_time_seconds"] = round(inference_time, 3)

        all_annotations.extend(annotations)

        if len(all_annotations) >= examples_to_create:
            break

    return all_annotations


def _create_prompt(
    generate_prompt: str, examples_to_create: int, language_iso: str
) -> str:
    return (
        generate_prompt
        + f"""
    
Generate {examples_to_create} data items depending on content richness. The content \
of the data items must be in the language with ISO code "{language_iso}".

JSON:"""
    )


def _create_prompt_with_text(
    text: str, generate_prompt: str, examples_to_create: int, language_iso: str
) -> str:
    return (
        generate_prompt
        + f"""
    
Generate {examples_to_create} data items depending on content richness. The content \
of the data items must be in the language with ISO code "{language_iso}".

TEXT:

{text}

JSON:"""
    )


def _parse_response(response: str) -> list[dict[str, Any]]:
    """Parse the model response and extract items."""
    json_start = response.find("{")
    json_end = response.rfind("}") + 1

    if json_start == -1 or json_end == 0:
        raise JsonNotFoundError("No JSON found in response")

    json_str = response[json_start:json_end]
    parsed = json.loads(json_str)
    items = parsed.get("items")
    if items is None:
        logger.warning("Field `items` not found in response: {response}")
        return []

    return items
