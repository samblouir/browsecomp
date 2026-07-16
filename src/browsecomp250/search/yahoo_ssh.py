from __future__ import annotations

import asyncio
import shlex
from urllib.parse import urlencode

from ..types import SearchResult
from .base import SearchError
from .yahoo import _CONTROL_TOKEN, YahooSearchProvider

_MAX_REMOTE_RESPONSE_BYTES = 4_000_000
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
)


class YahooSSHSearchProvider(YahooSearchProvider):
    """Fetch Yahoo organic results through an authorized SSH egress host."""

    name = "yahoo_ssh"

    async def _search_live(self, query: str, count: int, offset: int) -> list[SearchResult]:
        query = " ".join(_CONTROL_TOKEN.sub(" ", query).split())
        if not query:
            raise SearchError("Yahoo/SSH query was empty after control-token cleanup")
        if not self.config.yahoo_ssh_host.strip():
            raise SearchError("Yahoo/SSH requires yahoo_ssh_host")

        first = offset * count + 1
        target = f"{self.endpoint}?{urlencode({'p': query, 'b': first, 'pz': count})}"
        remote_command = shlex.join(
            [
                self.config.yahoo_ssh_remote_curl_bin,
                "--location",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time",
                str(self.config.timeout_seconds),
                "--user-agent",
                _USER_AGENT,
                "--header",
                "Accept: text/html,application/xhtml+xml",
                target,
            ]
        )

        async with self._request_semaphore:
            await self._wait_for_request_slot()
            process = await asyncio.create_subprocess_exec(
                self.config.yahoo_ssh_bin,
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.config.yahoo_ssh_connect_timeout_seconds}",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=1",
                self.config.yahoo_ssh_host,
                remote_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=(
                        self.config.timeout_seconds
                        + self.config.yahoo_ssh_connect_timeout_seconds
                        + 5
                    ),
                )
            except TimeoutError as exc:
                process.kill()
                await process.wait()
                await self._extend_cooldown()
                raise SearchError("Yahoo/SSH request timed out") from exc

        if process.returncode != 0:
            await self._extend_cooldown()
            detail = stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise SearchError(
                f"Yahoo/SSH request failed with exit {process.returncode}: {detail}"
            )
        if len(stdout) > _MAX_REMOTE_RESPONSE_BYTES:
            raise SearchError("Yahoo/SSH response exceeded the 4 MB safety limit")

        document = stdout.decode("utf-8", errors="replace")
        return self._parse_html(document, count=count, offset=offset)
