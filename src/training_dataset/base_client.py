import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class BaseClient(ABC):
    """Abstract base class for all model clients."""

    def __init__(self):
        self.session = requests.Session()

    def make_request(
        self,
        url: str,
        data: Dict[str, Any] | None = None,
        method: str = "POST",
        api_key: str | None = None,
    ) -> Dict[str, Any]:
        """Make a request to the API.

        Args:
            data: Request data for POST requests
            method: HTTP method (GET or POST)
            api_key: Optional API key for Authorization header

        Returns:
            JSON response as dictionary

        Raises:
            requests.exceptions.RequestException: If the request fails
        """

        # Set default headers
        request_headers = {"Content-Type": "application/json"}
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"

        if method.upper() == "GET":
            response = self.session.get(url, headers=request_headers, timeout=300)
        else:
            response = self.session.post(
                url, json=data, headers=request_headers, timeout=300
            )

        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion from a prompt.

        Args:
            prompt: The input prompt for text generation
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        pass
