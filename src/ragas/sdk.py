"""
SDK module for interacting with the Ragas API service.
"""

import os
from functools import lru_cache

import requests

from ragas._version import __version__
from ragas.exceptions import UploadException

# endpoint for uploading results
RAGAS_API_URL = "https://api.ragas.io"
RAGAS_APP_URL = "https://app.ragas.io"
RAGAS_API_SOURCE = "ragas_py"


@lru_cache(maxsize=1)
def get_app_token() -> str:
    app_token = os.environ.get("RAGAS_APP_TOKEN")
    if app_token is None:
        raise ValueError("RAGAS_APP_TOKEN is not set")
    return app_token


def upload_packet(path: str, data_json_string: str, base_url: str = RAGAS_API_URL):
    app_token = get_app_token()
    response = requests.post(
        f"{base_url}/api/v1{path}",
        data=data_json_string,
        headers={
            "Content-Type": "application/json",
            "x-app-token": app_token,
            "x-source": RAGAS_API_SOURCE,
            "x-app-version": __version__,
        },
    )
    if response.status_code == 403:
        raise UploadException(
            status_code=response.status_code,
            message="AUTHENTICATION_ERROR: The app token is invalid. Please check your RAGAS_APP_TOKEN environment variable.",
        )
    return response
