"""Check failures that would invalidate a retrieval experiment."""

from itertools import pairwise

import pytest

from app.corpus import canonical_url, chunks, html_text, official_url


def test_page_offsets_are_lossless_and_cover_tail() -> None:
    text = "\n".join(f"строка{i}  дата-2026" for i in range(33))
    result = chunks(text, size=10, overlap=3)
    assert result[0].start == 0
    assert result[-1].end == len(text)
    for item in result:
        assert text[item.start : item.end] == item.text
        assert item.words <= 10
    for left, right in pairwise(result):
        assert left.end > right.start  # context overlap, no gap


@pytest.mark.parametrize("size,overlap", [(0, 0), (2, 2), (2, -1)])
def test_reject_invalid_chunking(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunks("текст", size, overlap)


def test_main_extraction_keeps_repeated_table_values() -> None:
    html = b"""<html><head><title>Rules</title></head><body>
    <nav>global links</nav><main><h1>Admissions</h1><script>private JSON</script>
    <table><tr><td>Math</td><td>60</td></tr><tr><td>Physics</td><td>60</td></tr></table>
    </main><footer>Footer</footer></body></html>"""
    title, text, tables, _, _ = html_text(html)
    assert title == "Rules"
    assert text.count("60") == 2
    assert tables == 1
    assert all(noise not in text for noise in ["global links", "private JSON", "Footer"])


@pytest.mark.parametrize(
    "url",
    [
        "https://abit.itmo.ru.evil.test/page/66",
        "https://abit.itmo.ru@evil.test/",
        "https://abit.itmo.ru/page/66?user=1",
        "http://abit.itmo.ru/page/66",
        "https://abitlk.itmo.ru/",
        "file:///etc/passwd",
    ],
)
def test_collection_cannot_leave_allowlist(url: str) -> None:
    assert not official_url(canonical_url(url))


def test_fragment_is_not_another_document() -> None:
    assert canonical_url("https://abit.itmo.ru/page/66#dates") == "https://abit.itmo.ru/page/66"
    assert chunks("") == []
