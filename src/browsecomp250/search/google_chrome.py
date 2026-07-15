from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import shlex
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from ..types import SearchResult
from .base import SearchError, SearchProvider

_RESULT_BEGIN = "__BC250_GOOGLE_CHROME_RESULT_BEGIN__"
_RESULT_END = "__BC250_GOOGLE_CHROME_RESULT_END__"
_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", flags=re.I)
_SKIP_LINES = {
    "About this result",
    "Read more",
    "Show more",
    "Web results",
    "Search Results",
    "Search results",
}


class GoogleChromeSearchProvider(SearchProvider):
    """Search Google in the user's existing Chrome without a managed profile."""

    name = "google_chrome"

    def _ssh_prefix(self) -> list[str]:
        return [
            self.config.google_chrome_ssh_bin,
            "-o",
            f"ConnectTimeout={self.config.google_chrome_connect_timeout_seconds}",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-tt",
            self.config.google_chrome_host,
        ]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._deploy_lock = asyncio.Lock()
        self._batch_lock = asyncio.Lock()
        self._remote_bridge_path: str | None = None
        self.last_batch_metadata: dict[str, Any] = {}

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        batches = await self._search_many_live([(query, count, offset)])
        first = batches[0]
        if isinstance(first, Exception):
            raise first
        return first

    async def search_many(
        self,
        queries: list[str],
        count: int | None = None,
        offset: int = 0,
    ) -> list[list[SearchResult] | Exception]:
        normalized = [" ".join(str(query).split()).strip() for query in queries]
        resolved_count = min(count or self.config.results_per_call, 20)
        output: list[list[SearchResult] | Exception | None] = [None] * len(normalized)
        misses: list[tuple[int, str, dict[str, Any]]] = []

        for index, query in enumerate(normalized):
            if not query:
                output[index] = SearchError("Search query is empty")
                continue
            request = self._cache_request(query, resolved_count, offset)
            if self.config.cache_mode in {"read", "readwrite"}:
                cached = self.cache.get(request)
                if cached is not None:
                    output[index] = [SearchResult(**item) for item in cached]
                    continue
                if self.config.cache_mode == "read":
                    output[index] = SearchError(
                        f"Read-only search cache miss for provider={self.name}, query={query!r}"
                    )
                    continue
            misses.append((index, query, request))

        max_fanout = self.config.google_chrome_max_fanout
        for start in range(0, len(misses), max_fanout):
            chunk = misses[start : start + max_fanout]
            live = await self._search_many_live(
                [(query, resolved_count, offset) for _, query, _ in chunk]
            )
            for (index, _query, request), result in zip(chunk, live, strict=True):
                output[index] = result
                if not isinstance(result, Exception) and self.config.cache_mode in {
                    "write",
                    "readwrite",
                    "refresh",
                }:
                    self.cache.put(request, [asdict(item) for item in result])

        return [
            item if item is not None else SearchError("Google Chrome search produced no result")
            for item in output
        ]

    async def _search_many_live(
        self,
        requests: list[tuple[str, int, int]],
    ) -> list[list[SearchResult] | Exception]:
        if not self.config.google_chrome_host:
            error = SearchError("search.google_chrome_host or BC250_GOOGLE_CHROME_HOST is required")
            return [error for _ in requests]

        last_error: Exception | None = None
        for attempt in range(self.config.google_chrome_max_retries + 1):
            try:
                async with self._batch_lock:
                    payload = self._build_payload(requests)
                    response = await self._invoke_bridge(payload)
                self.last_batch_metadata = {
                    key: response.get(key)
                    for key in (
                        "request_tag",
                        "pid",
                        "bridge_pid",
                        "self_activation_suppressed",
                        "load_parallel",
                        "closed_tabs",
                        "launch_seconds",
                        "load_wait_seconds",
                        "batch_seconds",
                    )
                }
                rows_by_tag = {
                    str(row.get("tag") or ""): row
                    for row in response.get("results") or []
                    if isinstance(row, dict)
                }
                output: list[list[SearchResult] | Exception] = []
                for entry, (query, count, offset) in zip(payload["entries"], requests, strict=True):
                    row = rows_by_tag.get(str(entry["tag"]))
                    if not row:
                        output.append(SearchError("Chrome bridge omitted a query result"))
                    elif row.get("error"):
                        output.append(SearchError(str(row["error"])))
                    else:
                        try:
                            output.append(
                                self._parse_page_text(
                                    query,
                                    str(row.get("text") or ""),
                                    count,
                                    offset,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 - isolate one fanout result
                            output.append(exc)
                return output
            except Exception as exc:  # noqa: BLE001 - wrapped as provider error
                last_error = exc
                if attempt < self.config.google_chrome_max_retries:
                    await asyncio.sleep(min(2**attempt, 3))
        error = SearchError(f"Google-in-user-Chrome search failed: {last_error}")
        return [error for _ in requests]

    def _build_payload(self, requests: list[tuple[str, int, int]]) -> dict[str, Any]:
        request_tag = f"frlbc250-{uuid.uuid4().hex[:16]}"
        entries: list[dict[str, str]] = []
        safe = "active" if self.config.safe_search in {"moderate", "strict"} else "off"
        for index, (query, count, offset) in enumerate(requests):
            tag = f"{request_tag}-{index}"
            params = {
                "q": query,
                "num": str(count),
                "start": str(offset * count),
                "hl": self.config.language,
                "gl": self.config.country,
                "safe": safe,
                "pws": "0",
            }
            entries.append(
                {
                    "query": query,
                    "tag": tag,
                    "url": f"https://www.google.com/search?{urlencode(params)}#frlbc250={tag}",
                }
            )
        return {
            "request_tag": request_tag,
            "bundle_id": self.config.google_chrome_bundle_id,
            "cua_driver": self.config.google_chrome_cua_driver,
            "timeout_seconds": self.config.google_chrome_timeout_seconds,
            "entries": entries,
        }

    async def _ensure_bridge(self) -> str:
        if self._remote_bridge_path is not None:
            return self._remote_bridge_path
        async with self._deploy_lock:
            if self._remote_bridge_path is not None:
                return self._remote_bridge_path
            source = Path(__file__).with_name("_google_chrome_bridge.py")
            source_bytes = source.read_bytes()
            digest = hashlib.sha256(source_bytes).hexdigest()[:16]
            remote = f"/tmp/browsecomp250-google-chrome-{digest}.py"
            probe = await asyncio.create_subprocess_exec(
                *self._ssh_prefix(),
                f"test -f {shlex.quote(remote)}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            probe_timeout = min(self.config.google_chrome_timeout_seconds, 10)
            try:
                _probe_stdout, probe_stderr = await asyncio.wait_for(
                    probe.communicate(), timeout=probe_timeout
                )
            except TimeoutError:
                probe.kill()
                await probe.wait()
                raise SearchError(
                    f"Google Chrome bridge probe exceeded {probe_timeout:.0f}s"
                ) from None
            if probe.returncode == 0:
                self._remote_bridge_path = remote
                return remote
            if probe.returncode == 255:
                detail = probe_stderr.decode("utf-8", "replace").strip()
                raise SearchError(f"Could not reach Google Chrome host: {detail}")

            encoded = base64.urlsafe_b64encode(source_bytes).decode("ascii")
            code = (
                "import base64,pathlib,sys;"
                "pathlib.Path(sys.argv[1]).write_bytes("
                "base64.urlsafe_b64decode(sys.argv[2]+'==='))"
            )
            remote_command = " ".join(
                (
                    shlex.quote(self.config.google_chrome_python_bin),
                    "-c",
                    shlex.quote(code),
                    shlex.quote(remote),
                    shlex.quote(encoded),
                )
            )
            process = await asyncio.create_subprocess_exec(
                *self._ssh_prefix(),
                remote_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stage_timeout = min(self.config.google_chrome_timeout_seconds, 20)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=stage_timeout
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                raise SearchError(
                    f"Google Chrome bridge staging exceeded {stage_timeout:.0f}s"
                ) from None
            if process.returncode != 0:
                detail = (stderr or stdout).decode("utf-8", "replace").strip()
                raise SearchError(f"Could not stage Google Chrome bridge: {detail}")
            self._remote_bridge_path = remote
            return remote

    async def _invoke_bridge(self, payload: dict[str, Any]) -> dict[str, Any]:
        remote = await self._ensure_bridge()
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        process = await asyncio.create_subprocess_exec(
            *self._ssh_prefix(),
            self.config.google_chrome_python_bin,
            remote,
            "--payload-base64",
            encoded,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timeout = self.config.google_chrome_timeout_seconds * max(
            2, len(payload.get("entries") or [])
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise SearchError(f"Google Chrome bridge exceeded {timeout:.0f}s") from None
        text = stdout.decode("utf-8", "replace").replace("\r", "")
        start = text.rfind(_RESULT_BEGIN)
        end = text.find(_RESULT_END, start + len(_RESULT_BEGIN)) if start >= 0 else -1
        if start < 0 or end < 0:
            detail = stderr.decode("utf-8", "replace").strip() or text[-1000:]
            raise SearchError(f"Google Chrome bridge returned no result envelope: {detail}")
        parsed = json.loads(text[start + len(_RESULT_BEGIN) : end].strip())
        if not isinstance(parsed, dict):
            raise SearchError("Google Chrome bridge returned a non-object")
        if not parsed.get("ok"):
            raise SearchError(str(parsed.get("error") or "Google Chrome bridge failed"))
        return parsed

    @classmethod
    def _parse_page_text(
        cls,
        query: str,
        text: str,
        count: int,
        offset: int,
    ) -> list[SearchResult]:
        lines = [line.strip() for line in text.replace("\r", "").splitlines() if line.strip()]
        for marker in ("Skip to main content", "Search Results", "Search results"):
            if marker in lines:
                lines = lines[lines.index(marker) + 1 :]
                break
        for marker in ("Tab Search", "Footer Links"):
            if marker in lines:
                lines = lines[: lines.index(marker)]
                break

        results: list[SearchResult] = []
        seen: set[str] = set()
        for index, line in enumerate(lines):
            match = _URL_RE.search(line)
            if match is None:
                continue
            if line.strip().rstrip("/") == match.group(0).strip().rstrip("/"):
                # Chrome exposes each result's display URL both inside the titled
                # link label and again as a standalone accessibility row.
                continue
            url = cls._display_url(line, match)
            parsed = urlsplit(url)
            host = parsed.netloc.lower().removeprefix("www.")
            if not host or host.endswith("google.com") or host.endswith("googleusercontent.com"):
                continue
            canonical = urlunsplit(
                (
                    parsed.scheme.lower(),
                    parsed.netloc.lower(),
                    parsed.path.rstrip("/"),
                    parsed.query,
                    "",
                )
            )
            if canonical in seen:
                continue
            seen.add(canonical)
            title = line[: match.start()].strip(" -|·") or host
            snippets: list[str] = []
            for following in lines[index + 1 : index + 8]:
                if _URL_RE.search(following):
                    break
                if following in _SKIP_LINES or following.startswith("About this result"):
                    continue
                if following == title or following.lower() == host:
                    continue
                snippets.append(following)
                if sum(len(part) for part in snippets) >= 500:
                    break
            results.append(
                SearchResult(
                    title=title[:500],
                    url=url,
                    snippet=" ".join(snippets)[:1200],
                    rank=offset * count + len(results) + 1,
                    source="google_user_chrome",
                )
            )
            if len(results) >= count:
                break
        if not results:
            lowered = text.lower()
            if "unusual traffic" in lowered or "captcha" in lowered:
                raise SearchError("Google presented a traffic challenge in the user's Chrome")
            raise SearchError(f"Google Chrome returned no parseable results for {query!r}")
        return results

    @staticmethod
    def _display_url(line: str, match: re.Match[str]) -> str:
        base = match.group(0).rstrip(".,;:)")
        tail = line[match.end() :].strip()
        if not tail.startswith("›"):
            return base
        segments: list[str] = []
        for raw in tail.split("›")[1:6]:
            value = raw.strip(" /·|,.;:")
            if not value or value.lower().startswith(("about this result", "translate")):
                break
            segments.append(quote(value, safe="-._~%"))
        return base.rstrip("/") + ("/" + "/".join(segments) if segments else "")


__all__ = ["GoogleChromeSearchProvider"]
