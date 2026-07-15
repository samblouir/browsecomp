# Known limitations

## Custom subset

BrowseComp-250 is not an official split. Its score cannot be substituted for the full 1,266-item BrowseComp score. Sampling uncertainty is materially larger, and the fixed subset may be easier or harder than the full set for a particular system.

## Static public benchmark

Models may have memorized benchmark questions, answers, or solution traces. A high score does not prove live research ability. Search engines may also index benchmark-derived content.

## Mutable web and search

Search rankings, pages, redirects, domain availability, geolocation, bot defenses, and provider APIs change. Two nominally identical runs can receive different evidence.

## Search-system dependence

BrowseComp evaluates a model-agent-search-browser system. Search provider quality and browser extraction may dominate some items. A score should not be described as pure parametric model knowledge.

## Semantic grader dependence

LLM grading can make mistakes and can drift behind mutable model aliases. The standard grader checks equivalence to a reference answer, not citation validity or explanation quality.

## No OCR

Image-only PDFs and text embedded solely in images are not OCRed. The direct browser does not interpret images, video, audio, interactive maps, or charts.

## Limited JavaScript interaction

Playwright renders pages but does not implement general clicking, scrolling, form completion, login, CAPTCHA solving, or stateful browsing workflows. It captures page HTML after DOM content load.

## No arbitrary computation tool

The agent cannot execute Python, shell commands, or spreadsheets. Some BrowseComp questions may benefit from structured computation. This restriction is intentional for safety and comparability but may lower attainable scores.

## Token counting

Token and cost reporting depends on endpoint `usage` fields. The harness does not locally tokenize prompts because custom OpenAI-compatible models may use unknown tokenizers. Hidden reasoning tokens may be absent or provider-specific.

## Timeout accounting

`AgentOutcome.duration_seconds` measures the model/agent phase. The outer task timeout includes the whole agent run but not necessarily grader latency. Campaign wall-clock time also includes concurrency scheduling and reporting overhead.

## Cache replay

Read-only caches reproduce only requests already present. Different models generate different search queries and URLs, so a cache warmed by one model is not a complete fixed web snapshot for another.

## Paired analysis still depends on protocol equivalence

`bc250 paired-compare` reports a paired bootstrap interval and exact McNemar p-value, but statistical output is meaningful only when the runs use the same subset and external scaffold. The command warns about detected protocol mismatches; it cannot detect every hidden provider-side difference.

## No hard dollar budget

The agent enforces turn, tool, character, and time budgets. It records estimated cost but does not stop mid-trial at a dollar threshold.

## Endpoint nondeterminism

Temperature zero does not guarantee deterministic outputs on distributed or quantized inference systems. Routing, batching, speculative decoding, floating-point reduction order, and mutable deployment aliases can change responses.
