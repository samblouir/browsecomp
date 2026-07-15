from __future__ import annotations

import io
import json
import re
from hashlib import sha256
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from markdownify import markdownify
from pypdf import PdfReader

from ..types import PageDocument
from ..util import truncate_middle, utc_now_iso

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", text).strip()


def extract_html(
    content: bytes,
    *,
    requested_url: str,
    final_url: str,
    status_code: int,
    content_type: str,
    max_links: int,
) -> PageDocument:
    soup = BeautifulSoup(content, "html.parser")
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    for element in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "template"]):
        element.decompose()

    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = urljoin(final_url, str(anchor.get("href", "")).strip())
        text = clean_text(anchor.get_text(" ", strip=True)) or href
        if href.startswith(("http://", "https://")) and href not in seen:
            seen.add(href)
            links.append({"text": text[:300], "url": href})
            if len(links) >= max_links:
                break

    container = soup.find("main") or soup.find("article") or soup.body or soup
    rendered = markdownify(str(container), heading_style="ATX", strip=["img"])
    text = clean_text(rendered)
    return PageDocument(
        requested_url=requested_url,
        final_url=final_url,
        title=title,
        text=text,
        content_type=content_type,
        status_code=status_code,
        links=links,
        fetched_at=utc_now_iso(),
        sha256=sha256(content).hexdigest(),
    )


def extract_pdf(
    content: bytes,
    *,
    requested_url: str,
    final_url: str,
    status_code: int,
    content_type: str,
) -> PageDocument:
    reader = PdfReader(io.BytesIO(content))
    parts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            text = f"[PDF page {page_number} extraction failed: {exc}]"
        parts.append(f"\n\n## Page {page_number}\n\n{text}")
    title = ""
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title)
    return PageDocument(
        requested_url=requested_url,
        final_url=final_url,
        title=title,
        text=clean_text("".join(parts)),
        content_type=content_type,
        status_code=status_code,
        fetched_at=utc_now_iso(),
        sha256=sha256(content).hexdigest(),
    )


def extract_document(
    content: bytes,
    *,
    requested_url: str,
    final_url: str,
    status_code: int,
    content_type: str,
    max_links: int,
) -> PageDocument:
    lowered = content_type.lower()
    if "pdf" in lowered or content.startswith(b"%PDF"):
        return extract_pdf(
            content,
            requested_url=requested_url,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
        )
    if (
        any(kind in lowered for kind in ("html", "xml", "xhtml"))
        or b"<html" in content[:1000].lower()
    ):
        return extract_html(
            content,
            requested_url=requested_url,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            max_links=max_links,
        )
    if "json" in lowered:
        try:
            parsed = json.loads(content.decode("utf-8", errors="replace"))
            text = json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            text = content.decode("utf-8", errors="replace")
    else:
        text = content.decode("utf-8", errors="replace")
    return PageDocument(
        requested_url=requested_url,
        final_url=final_url,
        title="",
        text=clean_text(text),
        content_type=content_type,
        status_code=status_code,
        fetched_at=utc_now_iso(),
        sha256=sha256(content).hexdigest(),
    )


def page_window(document: PageDocument, offset: int, max_chars: int) -> dict[str, object]:
    offset = max(0, offset)
    max_chars = max(1, max_chars)
    end = min(len(document.text), offset + max_chars)
    text = document.text[offset:end]
    return {
        "requested_url": document.requested_url,
        "final_url": document.final_url,
        "title": document.title,
        "content_type": document.content_type,
        "status_code": document.status_code,
        "offset": offset,
        "end_offset": end,
        "total_chars": len(document.text),
        "has_more": end < len(document.text),
        "text": truncate_middle(text, max_chars),
        "links": document.links,
        "sha256": document.sha256,
    }
