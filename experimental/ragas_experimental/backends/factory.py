"""Factory class for creating the backends or mocked backends."""

__all__ = ["RagasApiClientFactory"]

import os
import typing as t

from .ragas_api_client import RagasApiClient


class RagasApiClientFactory:
    """Factory for creating Ragas API client instances."""

    @staticmethod
    def create(
        app_token: t.Optional[str] = None,
        base_url: t.Optional[str] = None,
    ) -> RagasApiClient:
        """Create a Ragas API client.

        Args:
            api_key: The API key for the Ragas API
            base_url: The base URL for the Ragas API

        Returns:
            RagasApiClient: A Ragas API client instance
        """
        if app_token is None:
            app_token = os.getenv("RAGAS_APP_TOKEN")

        if app_token is None:
            raise ValueError("RAGAS_API_KEY environment variable is not set")

        if base_url is None:
            base_url = os.getenv("RAGAS_API_BASE_URL")

        if base_url is None:
            base_url = "https://api.dev.app.ragas.io"

        return RagasApiClient(app_token=app_token, base_url=base_url)
