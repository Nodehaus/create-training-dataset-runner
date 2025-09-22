import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from tqdm import tqdm

from .clients.base_client import BaseClient
from .utils import split_text_into_chunks_with_offsets

logger = logging.getLogger(__name__)


class JsonNotFoundError(Exception):
    """Raised when JSON response cannot be found or parsed from LLM output."""

    pass


@dataclass
class Obligation:
    """Represents a legal obligation."""

    type: str  # "requirement" or "prohibition"
    description: str
    scope_subject: str
    scope_affected_parties: str
    context: str


@dataclass
class QuestionAnswer:
    """Represents a question/answer pair."""

    question: str
    answer: str
    complexity: str  # "simple", "medium", "complex"
    category: str  # question category or topic


class BaseAnnotationGenerator(ABC):
    """Base class for annotation generators with common functionality."""

    input_field: str = "source_text"
    output_field: str = "annotation"

    @classmethod
    def generate_annotations(
        cls,
        llm_client: BaseClient,
        text: str,
        document_id: str,
        use_chunking: bool = True,
    ) -> List[Dict[str, Any]]:
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

        for chunk, start_offset, end_offset in tqdm(chunks_with_offsets):
            max_retries = 3
            retry_count = 0
            annotations = []

            while retry_count < max_retries:
                prompt = cls._create_prompt(chunk)
                start_time = time.time()
                response_text = llm_client.generate(prompt)
                end_time = time.time()
                inference_time = end_time - start_time

                try:
                    annotations = cls._parse_response(response_text)
                    break  # Success, exit retry loop
                except (JsonNotFoundError, json.decoder.JSONDecodeError):
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(
                            f"JSON parsing error, retrying "
                            f"({retry_count}/{max_retries}), "
                            f"response was: {response_text}"
                        )
                    else:
                        logger.error(
                            f"JSON parsing error after {max_retries} retries, "
                            f"skipping chunk, response was: {response_text}"
                        )

            for annotation in annotations:
                annotation["document_id"] = document_id
                annotation["within_start"] = start_offset
                annotation["within_end"] = end_offset
                annotation["inference_time_seconds"] = round(inference_time, 3)

            all_annotations.extend(annotations)

        return all_annotations

    @classmethod
    def generate_prompts(
        cls, text: str, use_chunking: bool = True
    ) -> List[tuple[str, int, int]]:
        """
        Generate prompts with chunking support.

        Args:
            text: The document text to analyze
            use_chunking: Whether to split text into chunks (default: True)

        Returns:
            List of (prompt, start_offset, end_offset) tuples
        """
        if use_chunking:
            chunks_with_offsets = split_text_into_chunks_with_offsets(
                text, max_chunk_size=4000, overlap=100
            )
        else:
            chunks_with_offsets = [(text, 0, len(text))]

        prompts_with_offsets = []
        for chunk, start_offset, end_offset in chunks_with_offsets:
            prompt = cls._create_prompt(chunk)
            prompts_with_offsets.append((prompt, start_offset, end_offset))

        return prompts_with_offsets

    @staticmethod
    @abstractmethod
    def _create_prompt(text: str) -> str:
        """Create prompt for the specific annotation type."""
        pass

    @staticmethod
    def _parse_response(response: str) -> List[Dict[str, Any]]:
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


class ObligationAnnotationGenerator(BaseAnnotationGenerator):
    """Generator for creating obligation annotations from legal text using Ollama."""

    input_field: str = "source_text"
    output_field: str = "annotation"

    @staticmethod
    def _create_prompt(text: str) -> str:
        """Create prompt for obligation extraction."""
        return f"""You are a legal AI assistant specializing in EU law. Extract legal \
obligations from the provided EUR-Lex text.

Focus on requirements (what must be done) and prohibitions (what must not be done).

Return your response as valid JSON with the following structure for each \
obligation found:
{{
  "items": [
    {{
      "type": "requirement|prohibition",
      "description": "Clear summary of what must/must not be done",
      "scope_subject": "Who is obligated, who must comply with the rules",
      "scope_affected_parties": "Those who need to be aware because the \
obligation impacts them, even if they're not directly obligated",
      "context": "When it applies (simplified conditions)"
    }}
  ]
}}

Only return valid JSON. If no clear obligations are found, return \
{{"items": []}}.

TEXT:

{text}

JSON:"""


class QuestionAnswerGenerator(BaseAnnotationGenerator):
    """Generator for creating question/answer pairs from legal text using Ollama."""

    input_field: str = "question"
    output_field: str = "answer"

    @staticmethod
    def _create_prompt(text: str) -> str:
        """Create prompt for question/answer generation."""
        text_length = len(text)
        # Calculate questions based on ~1 question per 400 characters, min 1
        max_questions = max(1, text_length // 400)
        min_questions = max(1, max_questions // 2)
        question_count = f"{min_questions}-{max_questions}"

        return f"""You are an AI assistant specializing in creating educational \
question/answer pairs from legal texts. Generate diverse questions and answers \
based on the provided EUR-Lex text.

Create questions of three complexity levels:
- Simple: Basic facts, definitions, key terms
- Medium: Relationships, implications, comparisons  
- Complex: Analysis, synthesis, evaluation, critical thinking

Generate {question_count} question/answer pairs depending on content richness. \
Ensure variety in question types and complexity levels.

Return your response as valid JSON with the following structure:
{{
  "items": [
    {{
      "question": "Clear, specific question about the text",
      "answer": "Complete, accurate answer based on the text",
      "complexity": "simple|medium|complex"
    }}
  ]
}}

Only return valid JSON. If the text doesn't contain enough information for \
meaningful questions, return {{"items": []}}.

TEXT:

{text}

JSON:"""
