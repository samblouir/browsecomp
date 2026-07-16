# Search and browser configuration

## Search adapters

Every adapter returns:

```python
SearchResult(
    title: str,
    url: str,
    snippet: str,
    rank: int,
    source: str,
    extra_snippets: list[str],
)
```

Provider-specific answer boxes, knowledge panels, and generated answers are not normalized into a privileged tool. Only the mapped organic result fields are exposed unless the adapter explicitly records extra snippets.

### Brave Search

Configuration:

```dotenv
BC250_SEARCH_PROVIDER=brave
BC250_BRAVE_API_KEY=...
```

The adapter uses the Brave Web Search endpoint and `X-Subscription-Token`. Country, language, safe-search, count, and offset are passed from the common configuration.

`brave_ssh` is a distinct, keyless stratum that parses Brave's server-rendered
organic result page through an authorized SSH egress host. It uses bounded and
paced `curl` commands, rejects challenge/empty pages, validates public URLs, and
filters result URLs that merely mirror the query. It never launches the user's
Chrome profile.

### Experimental Google in the user's Chrome

```dotenv
BC250_SEARCH_PROVIDER=google_chrome
BC250_GOOGLE_CHROME_HOST=sam-mbp-rev
```

This provider is retained for isolated development only and is not used by the
Star benchmark profiles. It uses the existing Google Chrome process and signed-in profile on
the selected Mac. It does not start a managed profile or read the profile
directory. `search_many` sends every query URL to CUA in one launch call, so
Google loads the pages concurrently in background tabs. Accessibility text is
then read from each exact tagged tab. A request-scoped tag and `finally` cleanup
ensure the bridge closes only tabs it created.

The remote bridge is hash-versioned under `/tmp`, bounded by SSH/connect/page
timeouts, and serialized across benchmark tasks with a remote lock while each
batch still fans out internally. If Google shows a challenge or the personal
Chrome host is unavailable, the provider returns a normal per-query error.

### Hybrid Google + Brave

```dotenv
BC250_SEARCH_PROVIDER=hybrid
BC250_GOOGLE_CHROME_HOST=sam-mbp-rev
BC250_BRAVE_API_KEY=...
```

`hybrid` is retained for isolated development only. It starts the Google batch and Brave API
queries concurrently, interleaves successful results, and deduplicates exact
URLs. One engine may fail without discarding the other engine's results. The
logical search budget counts requested queries, not the number of engines used;
reports must disclose hybrid search rather than calling it a Brave-only run.

Do not use either Chrome-backed provider for headline or development runs.

### OpenRouter Exa web search

```dotenv
BC250_SEARCH_PROVIDER=openrouter_exa
BC250_OPENROUTER_API_KEY=...
BC250_OPENROUTER_SEARCH_MODEL=openai/gpt-4.1-nano
BC250_OPENROUTER_SEARCH_MAX_CONCURRENCY=32
```

This adapter forces OpenRouter's `web` plugin to use Exa and requests ten
results per query. The configured carrier model is transport only: its generated
answer is discarded. Only standardized `url_citation` annotations are mapped to
titles, URLs, and snippets. The run manifest records live request count, cited
result count, carrier input/output tokens, provider-reported cost, and the number
of rejected query-mirroring results. Adapter version, carrier model, engine, and
maximum result count are part of the cache identity.

Exa annotations can contain article-scale citation text rather than a conventional
search snippet. The adapter bounds every citation passage to 2,400 characters,
preserving both its beginning and end, so one early result cannot hide later
results or dominate the agent context. The cache identity includes this limit,
and the manifest records both the number of truncated snippets and the number of
characters removed.

The adapter rejects a narrow class of retrieval-poisoning pages whose long
title and URL slug mechanically reproduce a long search query and whose snippet
adds little independent information. This is intentionally not a general domain
blocklist: a separately authored article is retained even when its title happens
to overlap the query. The model prompt also requires clue decomposition and
candidate-centric follow-up searches instead of answer-shaped copies of the
question.

The evaluated reasoning model remains Star-7, and `ask_external_model` remains
the separately configured Star-2 agent. Exa search therefore does not insert a
third model's answer into the research transcript. This paid provider requires a
working OpenRouter credential and is a distinct reporting stratum from Brave or
Yahoo.

### Server-side Yahoo

```dotenv
BC250_SEARCH_PROVIDER=yahoo
```

The direct Yahoo adapter requests the server-rendered result page, extracts
organic result titles, snippets, and links, and unwraps Yahoo tracking redirects
to their destination URLs. It does not use a personal browser and requires no
API credential. Provider HTML is not a stable API contract, so live doctor and
fanout smokes are required before a campaign starts.

`yahoo_jina` is retained as a diagnostic fallback through the public Jina
reader. Anonymous reader rate limits make it unsuitable as the primary provider
for a concurrent headline campaign.

`yahoo_ssh` uses the direct Yahoo parser through an explicitly authorized SSH
egress host. It does not open Chrome or create a managed browser profile. SSH
connect limits, remote request limits, response-size limits, pacing, and
concurrency are enforced by the adapter and recorded in the protocol lock.

`bing_ssh` uses Bing's server-rendered RSS result surface through the same kind
of explicitly authorized SSH egress. It rejects HTML challenge pages, malformed
feeds, doctypes, empty result sets, oversized responses, and non-HTTP result
links. Its host, pacing, concurrency, and timeout limits are frozen in the run
lock, so Bing and Yahoo rows remain separately attributable benchmark strata.

Star-2 research helpers retain the full configured 16,384-token generation
budget per turn. Their separate action, evidence, and elapsed-research limits
force a final tool turn instead of allowing exploratory browsing to consume the
outer helper timeout. If a transport or model call still reaches that timeout,
the broker returns the executed queries, observed source URLs, and latest
research notes as explicitly partial evidence rather than discarding the work.

The Star smoke, development, and headline profiles default to `brave` but honor
an explicit `BC250_SEARCH_PROVIDER` override. The resolved provider is frozen in
the run lock. Any provider change within a larger development campaign must be
reported as a separate stratum or with exact per-row provenance.

### Tavily

```dotenv
BC250_SEARCH_PROVIDER=tavily
BC250_TAVILY_API_KEY=...
```

The adapter sends a bearer token and requests standard search depth. Tavily's returned content is mapped to the common snippet field.

### Serper

```dotenv
BC250_SEARCH_PROVIDER=serper
BC250_SERPER_API_KEY=...
```

Organic results are mapped from title, link, snippet, and position.

### SearXNG

```dotenv
BC250_SEARCH_PROVIDER=searxng
BC250_SEARXNG_BASE_URL=http://127.0.0.1:8080
```

The SearXNG instance must permit `format=json`. Freeze its engine list, categories, language, safe-search, timeout, proxy, and version during a comparison campaign.

## Search caching

Modes:

| Mode | Read cache | Use live provider on miss | Write response |
|---|---:|---:|---:|
| `off` | No | Yes | No |
| `read` | Yes | **No; fail closed** | No |
| `write` | No | Yes | Yes |
| `readwrite` | Yes | Yes | Yes |
| `refresh` | No | Yes | Yes/replace |

Cache keys include provider, normalized query, count, offset, country, language, and safe-search setting.

A read-only cache is useful for replaying an identical trajectory, but it is not generally sufficient for cross-model comparison: different models generate different queries, so one model may encounter cache misses where another does not.

## Direct browser

The direct backend:

1. validates the requested URL;
2. resolves the hostname and rejects blocked address classes;
3. issues an HTTP request without automatic redirects;
4. validates each redirect destination before requesting it;
5. enforces redirect and byte limits;
6. extracts HTML, PDF, JSON, or text; and
7. records final URL, status, content type, title, links, fetch timestamp, and content SHA-256.

HTML extraction removes scripts, styles, frames, canvases, templates, and SVG, then converts the main/article/body region to normalized Markdown.

PDF extraction uses text extraction only. Scanned image-only PDFs are not OCRed.

### Public reader fallback

The Star profiles enable a public text-reader fallback for public pages that
fail direct retrieval or yield fewer than 300 extracted characters. The
original URL is validated by the same private-network policy before any reader
request. If an HTTPS origin is unavailable but the validated origin is
HTTP-only, the fallback may retry that same host over HTTP.

The returned document keeps the original or resolved origin as its citation
URL. The reader URL is transport metadata, never source identity. Reader mode,
base URL, and minimum-character threshold are included in the page-cache key
and must be frozen for comparisons.

## Playwright browser

Install:

```bash
.venv/bin/pip install -e '.[browser]'
.venv/bin/playwright install chromium
```

Set:

```yaml
browser:
  backend: playwright
```

Playwright is useful for JavaScript-rendered sites. The harness intercepts page and subresource requests and aborts destinations that violate the network policy.

The browser does not persist cookies, accounts, extensions, or a human profile. Sites requiring login, CAPTCHA, geofenced access, or complex consent interaction may remain inaccessible.

## Auto backend

`auto` first uses direct HTTP. If extracted HTML text is extremely sparse, it attempts Playwright. This adaptive behavior can improve coverage but introduces a conditional browser difference. Fix it across all compared systems.

## Browser caching

Browser cache modes follow the same table as search caching. The page cache key includes URL, backend, response limit, link limit, and user agent.

Cached entries contain page text and URLs. Treat the cache as private evaluation data because its contents can reveal model search paths and benchmark clues.

## Network policy

Blocked by default:

- loopback addresses;
- RFC1918/private addresses;
- link-local addresses;
- multicast addresses;
- reserved and unspecified addresses;
- localhost and `.local` hostnames;
- embedded URL credentials;
- non-HTTP(S) navigation; and
- nonstandard ports.

Do not disable `block_private_networks` on a host that can reach cloud metadata endpoints, corporate networks, production services, or local developer systems.

## Fair-comparison checklist

Freeze:

- search provider and subscription tier;
- provider region and language;
- safe-search mode;
- number of results;
- browser backend and version;
- user agent;
- concurrency;
- page and search caches;
- evaluation window;
- budgets and timeouts; and
- outbound network routing/proxy.

Record search/provider outages separately from model failures, but count affected trials as incorrect in the headline denominator.
