# Security policy

## Scope

This project executes an autonomous model with access to web search and page retrieval. Treat every endpoint, page, search result, model output, and benchmark artifact as untrusted.

## Default controls

- Only `http` and `https` page URLs are accepted.
- URL credentials are rejected.
- Nonstandard ports are rejected unless explicitly enabled.
- Localhost and `.local` hostnames are blocked.
- Literal and DNS-resolved private, loopback, link-local, multicast, reserved, and unspecified addresses are blocked.
- Redirect destinations are validated before each request.
- Playwright subresources are routed through the same network policy.
- Response size, redirect count, page text, link count, search calls, page opens, retrieved characters, model steps, and task duration are bounded.
- API secrets are loaded from environment variables and redacted from run locks.
- Private transcripts can be encrypted with Fernet.
- Public artifacts are scanned against decrypted benchmark questions and answers before release.

## Operator responsibilities

Run the evaluator in an isolated account or machine with no privileged cloud metadata access, internal network routes, production credentials, SSH agents, browser profiles, or sensitive mounted directories. Use narrowly scoped search and model API credentials. Do not disable private-network blocking merely to resolve a website compatibility issue.

The direct browser is not a malware sandbox. It parses untrusted HTML and PDFs but does not execute downloaded binaries. The optional Playwright browser executes website JavaScript and should run in a disposable container or VM.

## Reporting vulnerabilities

Do not include BrowseComp plaintext questions, answers, canaries, API keys, private run artifacts, or customer data in a public issue. Report sensitive findings privately to the repository owner.
