# Third-party notices

This repository interoperates with the **BrowseComp** benchmark released in
OpenAI's `simple-evals` repository under the MIT License. It does not redistribute
the benchmark CSV or plaintext benchmark questions/answers. At runtime it downloads
the encrypted official CSV from the URL used by the OpenAI reference implementation.

The query and grader templates in `src/browsecomp250/prompts.py` are adapted from
OpenAI's MIT-licensed `browsecomp_eval.py` with attribution retained in source.

External services supported by adapters—Brave Search API, Tavily, Serper, SearXNG,
and any OpenAI-compatible model endpoint—remain subject to their own terms.
