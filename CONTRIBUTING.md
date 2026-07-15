# Contributing

Changes must preserve the distinction between the official 1,266-item BrowseComp benchmark and the custom BrowseComp-250 subset.

Before submitting a change:

```bash
python -m compileall -q src tests
pytest
ruff check src tests
ruff format --check src tests
```

Protocol-affecting changes require:

1. a version increment;
2. a changelog entry;
3. updated tests;
4. updated `docs/PROTOCOL.md`;
5. an explanation of whether existing scores remain comparable; and
6. regeneration of `configs/schema.json` and `FILE_SHA256SUMS.txt`.

Never commit:

- the official BrowseComp CSV;
- decrypted questions or answers;
- run transcripts;
- model/search/grader API credentials;
- search/page cache contents; or
- public examples derived from benchmark items.
