from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


class UnsafeURLError(ValueError):
    pass


def _is_disallowed_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def validate_url_syntax(
    url: str,
    *,
    block_private_networks: bool = True,
    allow_nonstandard_ports: bool = False,
) -> tuple[str, int | None]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeURLError(f"Unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise UnsafeURLError("URL has no hostname")
    if parsed.username or parsed.password:
        raise UnsafeURLError("Credentials in URLs are not permitted")
    host = parsed.hostname.rstrip(".").lower()
    if block_private_networks and (
        host in {"localhost", "localhost.localdomain"} or host.endswith(".local")
    ):
        raise UnsafeURLError(f"Local hostname is blocked: {host}")
    port = parsed.port
    if port is not None and not allow_nonstandard_ports and port not in {80, 443}:
        raise UnsafeURLError(f"Nonstandard port is blocked: {port}")
    if block_private_networks:
        try:
            is_disallowed_literal = _is_disallowed_ip(host)
        except ValueError:
            is_disallowed_literal = False
        if is_disallowed_literal:
            raise UnsafeURLError(f"Private or reserved IP is blocked: {host}")
    return host, port


async def assert_safe_url(
    url: str,
    *,
    block_private_networks: bool = True,
    allow_nonstandard_ports: bool = False,
    resolve_dns: bool = True,
) -> None:
    host, port = validate_url_syntax(
        url,
        block_private_networks=block_private_networks,
        allow_nonstandard_ports=allow_nonstandard_ports,
    )
    if not block_private_networks or not resolve_dns:
        return

    def resolve() -> list[str]:
        records = socket.getaddrinfo(host, port or 443, type=socket.SOCK_STREAM)
        return sorted({record[4][0] for record in records})

    try:
        addresses = await asyncio.to_thread(resolve)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"DNS resolution failed for {host}: {exc}") from exc
    if not addresses:
        raise UnsafeURLError(f"DNS resolution returned no addresses for {host}")
    blocked = [address for address in addresses if _is_disallowed_ip(address)]
    if blocked:
        raise UnsafeURLError(f"Hostname resolves to blocked address(es): {blocked}")
