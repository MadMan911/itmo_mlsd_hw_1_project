from pathlib import Path

import httpx
import pytest
from scripts.collect_corpus import Fetcher


def test_file_storage_is_not_opened_by_generic_pdf_allow(tmp_path: Path) -> None:
    fetcher = Fetcher(tmp_path)
    fetcher.parse_robots("User-agent: *\nDisallow: /file_storage/\nAllow: */*.pdf\n")
    assert not fetcher.allowed("https://abit.itmo.ru/file_storage/test.pdf")
    assert fetcher.allowed("https://abit.itmo.ru/page/66")
    fetcher.client.close()


def test_redirect_is_checked_before_second_request(tmp_path: Path) -> None:
    fetcher = Fetcher(tmp_path)
    fetcher.parse_robots("User-agent: *\nDisallow: /api/\n")
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "/api/secret"})

    fetcher.client.close()
    fetcher.client = httpx.Client(transport=httpx.MockTransport(respond))
    with pytest.raises(ValueError, match="disallows"):
        fetcher.get("https://abit.itmo.ru/page/66")
    assert calls == ["https://abit.itmo.ru/page/66"]
    fetcher.client.close()


def test_gzip_response_is_decoded_exactly_once(tmp_path: Path) -> None:
    import gzip

    fetcher = Fetcher(tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=gzip.compress(b"source text"), headers={"content-encoding": "gzip"}
        )

    fetcher.client.close()
    fetcher.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert fetcher.get("https://abit.itmo.ru/page/66").content == b"source text"
    fetcher.client.close()
