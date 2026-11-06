import pytest
import httpx
from findex.crawler import canonicalize_url, extract_links_and_text, RobotsCache, fetch_page


def test_canonicalize_url():
    url = "HTTPS://Example.COM:443/docs/intro/?b=2&a=1#section"
    expected = "https://example.com/docs/intro?a=1&b=2"
    assert canonicalize_url(url) == expected


def test_extract_links_and_text():
    html = """
    <html>
        <head><title>Test</title></head>
        <body>
            <nav><a href="/ignore">Skip</a></nav>
            <p>Привіт <b>світ</b></p>
            <a href="/docs/page1">Page 1</a>
            <a href="https://external.com/test#anchor">External</a>
        </body>
    </html>
    """
    text, links = extract_links_and_text("https://example.com/start", html)
    assert "Привіт світ" in text
    assert "Skip" not in text
    assert "https://example.com/docs/page1" in links
    assert "https://external.com/test" in links


@pytest.mark.asyncio
async def test_robots_cache_disallow():
    robots_content = "User-agent: *\nDisallow: /private/\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots_content)
        return httpx.Response(200, text="OK")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cache = RobotsCache(client, "test-bot")
        assert await cache.can_fetch("https://example.com/public/page") is True
        assert await cache.can_fetch("https://example.com/private/secret") is False


@pytest.mark.asyncio
async def test_fetch_page_retry_429():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text="Success")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        status, content, _ = await fetch_page(client, "https://example.com", max_retries=2)
        assert status == 200
        assert content == b"Success"
        assert attempts == 2