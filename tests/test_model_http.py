from urllib.error import HTTPError

from minicode_harness.models.http import _http_provider_error


def test_http_provider_error_classifies_qwen_arrearage_as_billing() -> None:
    error = HTTPError(
        url="https://dashscope.example/chat/completions",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=None,
    )
    body = (
        '{"error":{"message":"Access denied, see overdue-payment",'
        '"type":"Arrearage","code":"Arrearage"}}'
    )

    classified = _http_provider_error(error, body)

    assert classified.kind == "billing"
    assert classified.status_code == 400
    assert classified.retryable is False
