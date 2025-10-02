import logging
import os

from dotenv import load_dotenv

from training_dataset.runpod_client import RunpodClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")


def main():
    """Test inference with RunPod client."""
    # Initialize RunPod client
    client = RunpodClient(
        pod_id=RUNPOD_POD_ID,
        api_key=RUNPOD_API_KEY,
        model="qwen3:30b-a3b-instruct-2507-q4_K_M",
    )

    # If the input refers to source texts/corpus/context you will not mention them in your fields.

    # Test prompt
    prompt = """Tell me what list of JSON objects an LLM would output for a given prompt. \
It does not need to output any source text or reference, **ignore** any mentions of source text. \
Include minimum 2 fields, maximum 4 fields. We want to use the output data for LLM training. \
Decide which fields we will use as input and output of the training. \
Also return the expected size of the output in characters based on your output field description. \
Use the following output format:

```json
{
  "json_object_fields": {
    "field_name": "Description of what this field represents",
    "another_field": "Description of this field",
    "yet_another_field": "More descriptions"
  },
  "input_field": "field_name",
  "output_field": "another_field",
  "expected_output_size_chars": a_reasonable_int_for_output_size
}
```

PROMPT:

You are an AI assistant specializing in creating educational question/answer pairs from legal texts. Generate diverse questions and answers based on the provided EUR-Lex text.

Create questions of three complexity levels:
- Simple: Basic facts, definitions, key terms
- Medium: Relationships, implications, comparisons
- Complex: Analysis, synthesis, evaluation, critical thinking

Ensure variety in question types and complexity levels.

JSON:"""

    logger.info(f"Sending prompt:\n{prompt}\n\n")

    # Generate response
    response = client.generate(
        prompt=prompt,
        temperature=0.1,
        top_p=0.9,
        max_tokens=4096,
    )

    logger.info(f"RESPONSE:\n{response}")


if __name__ == "__main__":
    main()
