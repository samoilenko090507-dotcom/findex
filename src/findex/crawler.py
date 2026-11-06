import asyncio
import json
import logging
import random
import time
import typing
import urllib.robotparser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx
from selectolax.lexbor import LexborHTMLParser

logger = logging.getLogger("findex.crawler")

def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower() if parsed.hostname else ''
    port = parsed.port
    if port and not ((scheme == 'http' and port == 80) or (scheme == 'https' and port == 443)):
        netloc = f'{host}:{port}'
    else:
        netloc = host
    path = parsed.path or '/'
    if path != '/' and path.endswith('/'):
        path = path.rstrip('/')
    query_params = sorted(parse_qsl(parsed.query))
    query = urlencode(query_params)
    return urlunparse((scheme, netloc, path, parsed.params, query, ''))

class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str):
        self.client = client
        self.user_agent = user_agent
        self.cache: dict[str, urllib.robotparser.RobotFileParser | bool] = {}
        self.lock = asyncio.Lock()

    async def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        host_base = f"{parsed.scheme}://{parsed.netloc}"
        async with self.lock:
            if host_base not in self.cache:
                robots_url = f"{host_base}/robots.txt"
                try:
                    resp = await self.client.get(robots_url, timeout=5.0)
                    if resp.status_code == 200:
                        rp = urllib.robotparser.RobotFileParser()
                        rp.parse(resp.text.splitlines())
                        self.cache[host_base] = rp
                    else:
                        self.cache[host_base] = True
                except Exception:
                    self.cache[host_base] = True

            checker = self.cache[host_base]
            if isinstance(checker, bool):
                return checker
            return checker.can_fetch(self.user_agent, url)


def extract_links_and_text(base_url: str, html: str) -> tuple[str, list[str]]:
    tree = LexborHTMLParser(html)
    for tag in tree.css("script, style, noscript, nav, footer, header"):
        tag.decompose()

    body = tree.body
    text = body.text(separator=" ", strip=True) if body else ""

    links: list[str] = []
    for a in tree.css("a[href]"):
        raw_href = a.attributes.get("href")
        if not raw_href:
            continue
        try:
            absolute_url = urljoin(base_url, raw_href)
            parsed = urlparse(absolute_url)
            if parsed.scheme in ("http", "https"):
                links.append(canonicalize_url(absolute_url))
        except Exception:
            continue

    return text, links


async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    max_retries: int = 3,
    timeout_seconds: float = 10.0,
) -> tuple[int, bytes, float]:
    start_time = time.perf_counter()
    retries = 0

    while True:
        try:
            async with asyncio.timeout(timeout_seconds):
                resp = await client.get(url, follow_redirects=True)
                elapsed = time.perf_counter() - start_time

                if resp.status_code in (429, 500, 502, 503, 504):
                    if retries >= max_retries:
                        return resp.status_code, resp.content, elapsed

                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        sleep_time = float(retry_after)
                    else:
                        sleep_time = (2**retries) + random.uniform(0.1, 0.5)

                    retries += 1
                    await asyncio.sleep(sleep_time)
                    continue

                return resp.status_code, resp.content, elapsed

        except (httpx.RequestError, TimeoutError):
            elapsed = time.perf_counter() - start_time
            if retries >= max_retries:
                return 0, b"", elapsed

            sleep_time = (2**retries) + random.uniform(0.1, 0.5)
            retries += 1
            await asyncio.sleep(sleep_time)


class CrawlMetrics:
    def __init__(self) -> None:
        self.crawled: int = 0
        self.errors: int = 0
        self.active_requests: int = 0
        self.start_time: float = time.perf_counter()


async def crawl(
    start_urls: list[str],
    max_pages: int = 200,
    max_concurrency: int = 5,
    per_host_delay: float = 0.5,
    user_agent: str = "findex-bot/1.0 (+https://github.com/samoilenko090507-dotcom/findex)",
    metrics: CrawlMetrics | None = None,
) -> typing.AsyncGenerator[dict[str, str], None]:
    if metrics is None:
        metrics = CrawlMetrics()

    queue: asyncio.Queue[str] = asyncio.Queue()
    output_queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()
    seen: set[str] = set()

    for url in start_urls:
        canon = canonicalize_url(url)
        seen.add(canon)
        await queue.put(canon)

    global_semaphore = asyncio.Semaphore(max_concurrency)
    host_locks: dict[str, asyncio.Lock] = {}
    host_last_access: dict[str, float] = {}

    log_file = open("crawl.log", "a", encoding="utf-8")

    async def log_entry(status: int, url: str, elapsed: float, size: int) -> None:
        log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {status} {url} {size}B {elapsed:.3f}s\n")
        log_file.flush()

    stop_event = asyncio.Event()

    async def worker(client: httpx.AsyncClient, robots: RobotsCache) -> None:
        while not stop_event.is_set():
            try:
                url = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                if queue.empty() and metrics.active_requests == 0:
                    break
                continue

            parsed = urlparse(url)
            host = parsed.netloc

            if host not in host_locks:
                host_locks[host] = asyncio.Lock()

            allowed = await robots.can_fetch(url)
            if not allowed:
                await log_entry(403, url, 0.0, 0)
                metrics.errors += 1
                queue.task_done()
                continue

            metrics.active_requests += 1
            try:
                async with global_semaphore:
                    async with host_locks[host]:
                        now = time.monotonic()
                        last = host_last_access.get(host, 0.0)
                        wait_needed = per_host_delay - (now - last)
                        if wait_needed > 0:
                            await asyncio.sleep(wait_needed)
                        host_last_access[host] = time.monotonic()

                        status, content, elapsed = await fetch_page(client, url)

                await log_entry(status, url, elapsed, len(content))

                if status == 200:
                    metrics.crawled += 1
                    html = content.decode("utf-8", errors="replace")
                    text, links = extract_links_and_text(url, html)

                    await output_queue.put({"url": url, "text": text})

                    if metrics.crawled >= max_pages:
                        stop_event.set()

                    if not stop_event.is_set():
                        for link in links:
                            if link not in seen:
                                seen.add(link)
                                await queue.put(link)
                else:
                    metrics.errors += 1
            finally:
                metrics.active_requests -= 1
                queue.task_done()

    async def background_crawl() -> None:
        limits = httpx.Limits(max_keepalive_connections=max_concurrency, max_connections=max_concurrency * 2)
        async with httpx.AsyncClient(headers={"User-Agent": user_agent}, limits=limits) as client:
            robots = RobotsCache(client, user_agent)
            try:
                async with asyncio.TaskGroup() as tg:
                    for _ in range(max_concurrency):
                        tg.create_task(worker(client, robots))
            except Exception:
                pass
            finally:
                log_file.close()
                await output_queue.put(None)

    bg_task = asyncio.create_task(background_crawl())

    while True:
        item = await output_queue.get()
        if item is None:
            break
        yield item

    await bg_task

    import json
from rich.live import Live
from rich.table import Table


async def run_crawler_cli(
    start_url: str,
    output_path: str = "crawled_pages.jsonl",
    max_pages: int = 200,
    concurrency: int = 5,
    delay: float = 0.5,
) -> None:
    metrics = CrawlMetrics()

    def make_table() -> Table:
        table = Table(title="findex crawler status", border_style="cyan")
        table.add_column("Метрика", style="bold white")
        table.add_column("Значення", style="green")
        elapsed = max(time.perf_counter() - metrics.start_time, 0.001)
        speed = metrics.crawled / elapsed
        table.add_row("Зібрано сторінок", str(metrics.crawled))
        table.add_row("Швидкість (стор./с)", f"{speed:.2f}")
        table.add_row("Активні запити", str(metrics.active_requests))
        table.add_row("Помилки / Відхилено", str(metrics.errors))
        table.add_row("Час роботи", f"{elapsed:.1f}s")
        return table

    with open(output_path, "w", encoding="utf-8") as f_out, Live(
        make_table(), refresh_per_second=4
    ) as live:
        async for doc in crawl(
            start_urls=[start_url],
            max_pages=max_pages,
            max_concurrency=concurrency,
            per_host_delay=delay,
            metrics=metrics,
        ):
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            f_out.flush()
            live.update(make_table())
        live.update(make_table())