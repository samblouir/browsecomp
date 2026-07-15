#!/usr/bin/env python3
"""Tiny local OpenAI-compatible server for transport/preflight testing only.

It is not a model and must never be used to produce benchmark scores.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    server_version = "BC250Mock/0.1"

    def _json(self, status: int, value: object) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": "mock-model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        messages = request.get("messages") or []
        text = "\n".join(
            str(item.get("content", "")) for item in messages if isinstance(item, dict)
        )
        if "[correct_answer]" in text or "Judge whether" in text:
            content = (
                "extracted_final_answer: mock-answer\n"
                "reasoning: Mock server always grades incorrect.\n"
                "correct: no\nconfidence: 100"
            )
        elif "preflight" in text.casefold():
            content = '{"action":"note","text":"preflight"}'
        else:
            content = (
                '{"action":"final","explanation":"mock transport response",'
                '"exact_answer":"mock-answer","confidence":0,"citations":[]}'
            )
        self._json(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": request.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[mock] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    print(f"Mock OpenAI-compatible API: http://{args.host}:{args.port}/v1")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
