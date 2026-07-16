from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class SharedHostThrottle:
    semaphore: asyncio.Semaphore
    pace_lock: asyncio.Lock
    max_concurrency: int
    next_request_at: float = 0.0


_THROTTLES: dict[tuple[int, str, str], SharedHostThrottle] = {}


def shared_host_throttle(
    *,
    namespace: str,
    host: str,
    max_concurrency: int,
) -> SharedHostThrottle:
    """Return one process-wide limiter for an engine and egress host.

    Guided training creates one search provider per worker. Keeping the limiter
    on each provider turns nominally conservative settings into a burst of
    ``workers * max_concurrency`` requests from one IP address.
    """

    loop_id = id(asyncio.get_running_loop())
    key = (loop_id, namespace.strip().casefold(), host.strip().casefold())
    throttle = _THROTTLES.get(key)
    if throttle is None:
        throttle = SharedHostThrottle(
            semaphore=asyncio.Semaphore(max_concurrency),
            pace_lock=asyncio.Lock(),
            max_concurrency=max_concurrency,
        )
        _THROTTLES[key] = throttle
    return throttle


__all__ = ["SharedHostThrottle", "shared_host_throttle"]
