"""Encryption helpers compatible with OpenAI's BrowseComp reference code."""

from __future__ import annotations

import base64
import hashlib


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(left ^ right for left, right in zip(encrypted, key, strict=True))
    return decrypted.decode("utf-8")


def encrypt(plaintext: str, password: str) -> str:
    raw = plaintext.encode("utf-8")
    key = derive_key(password, len(raw))
    encrypted = bytes(left ^ right for left, right in zip(raw, key, strict=True))
    return base64.b64encode(encrypted).decode("ascii")
