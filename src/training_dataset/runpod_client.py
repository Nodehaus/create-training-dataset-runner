import logging

from .base_client import BaseClient

logger = logging.getLogger(__name__)

RUNPOD_BASE_URL = "https://api.runpod.ai"
MAX_TOKENS = 16384


class RunpodClient(BaseClient):
    """Client for interacting with RunPod IO API."""

    def __init__(self, pod_id: str, api_key: str, model: str):
        """Initialize RunPod client.

        Args:
            url: The RunPod endpoint URL
            pod_id: The Pod ID on RunPod
            api_key: Optional API key for Authorization header
        """
        super().__init__()
        if not pod_id:
            raise ValueError("You must specify a Pod ID for runpod.")
        self.pod_id = pod_id
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Generate text completion using RunPod API.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        payload = {
            "input": {
                "openai_route": "/v1/completions",
                "openai_input": {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
        }

        response = self.make_request(
            f"{RUNPOD_BASE_URL}/v2/{self.pod_id}/runsync", payload, api_key=self.api_key
        )
        try:
            response_text = response["output"][0]["choices"][0]["text"].strip()
        except KeyError:
            raise ValueError(
                f"Did not receive valid response from Runpod, response was: {response}"
            )
        return response_text
