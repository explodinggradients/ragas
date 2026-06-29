from unittest.mock import patch

from ragas.metrics.collections.multi_modal_faithfulness.util import (
    is_image_path_or_url,
    process_image_to_base64,
)
from ragas.prompt.multi_modal_prompt import ImageTextPromptValue


def test_image_url_rejects_backslash_userinfo_ssrf_bypass():
    prompt_value = ImageTextPromptValue(items=[])
    bypass_url = "http://127.0.0.1:6666\\@1.1.1.1/image.png"

    with patch("ragas.prompt.multi_modal_prompt.requests.get") as mock_get:
        assert prompt_value._download_validate_and_encode(bypass_url) is None

    mock_get.assert_not_called()


def test_image_url_rejects_userinfo_before_request():
    prompt_value = ImageTextPromptValue(items=[])

    with patch("ragas.prompt.multi_modal_prompt.requests.get") as mock_get:
        assert (
            prompt_value._try_process_allowed_url("https://user@example.com/a.png")
            is None
        )

    mock_get.assert_not_called()


def test_image_url_accepts_plain_http_url_shape():
    prompt_value = ImageTextPromptValue(items=[])

    assert prompt_value._is_allowed_image_url("https://example.com/a.png") is True


def test_collections_image_url_rejects_backslash_userinfo_ssrf_bypass():
    bypass_url = "http://127.0.0.1:6666\\@1.1.1.1/image.png"

    assert is_image_path_or_url(bypass_url) is False
    with patch(
        "ragas.metrics.collections.multi_modal_faithfulness.util.requests.get"
    ) as mock_get:
        assert process_image_to_base64(bypass_url) is None

    mock_get.assert_not_called()


def test_collections_image_url_blocks_internal_resolved_target():
    with (
        patch(
            "ragas.metrics.collections.multi_modal_faithfulness.util.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("127.0.0.1", 0))],
        ),
        patch(
            "ragas.metrics.collections.multi_modal_faithfulness.util.requests.get"
        ) as mock_get,
    ):
        assert process_image_to_base64("https://example.com/a.png") is None

    mock_get.assert_not_called()
