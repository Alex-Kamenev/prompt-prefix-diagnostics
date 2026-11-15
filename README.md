# prompt-prefix-diagnostics

Compare a JSON-level fingerprint of leading system messages and tool schemas
across requests belonging to an explicitly identified conversation and model.
This is a small standard-library utility, with no network or model integration.

Released under the [MIT License](LICENSE).

## Origin

Adapted from personal local developer tooling, then isolated and revised with
AI assistance for this release. This standalone adaptation was prepared in
November 2026.

## Run

Requires Python 3.10 or newer. From this directory:

```text
python example.py
python -m unittest discover -s tests -v
```

The example uses only synthetic values and prints `new`, `stable`, then
`changed`, each followed by a digest. No installation is needed.

```python
from prefix_diagnostics import PrefixTracker

tracker = PrefixTracker(max_tracked=128)
report = tracker.observe(
    conversation_id="example-conversation",
    model="example-model",
    messages=[{"role": "system", "content": "Use plain language."},
              {"role": "user", "content": "A synthetic question"}],
    tools=[],
)
print(report.status)
```

## Meaning of a result

- `new`: this exact `(conversation_id, model)` pair is not currently remembered.
- `stable`: its selected-field fingerprint matches its last valid observation.
- `changed`: its selected-field fingerprint differs from its last valid observation.

Conversation identity is supplied by the caller, not guessed from message counts
or text. Reuse an ID only for the same conversation and distinguish model
versions when relevant. IDs and model strings are case-sensitive, nonempty,
valid UTF-8, and limited to 256 encoded bytes each. Tuple keys prevent delimiter
collisions. Callers must provide trustworthy IDs; this is not an authentication
or tenant-isolation mechanism.

## What is fingerprinted

Only the consecutive messages whose role is exactly `system` at the beginning
of `messages`, plus the complete supplied tool list. The first other role ends
that run. Later `system` messages and `developer`, user, assistant or tool
messages are not fingerprinted. Changes to those parts can therefore produce
`stable`. Every field of a selected system-message object is included, even if
a provider would ignore that field. `None` tools and an empty tool list are
treated alike.

Inputs use built-in JSON types: objects with string keys, arrays, strings,
finite numbers, booleans and null. Each message needs a string role. Unsupported
objects, tuple values, non-finite floats, cyclic structures and excessive
nesting/node counts raise generic errors; there is no `str`/`repr` fallback.
Object key order is ignored; array order and exact string code points matter.
Strings are not Unicode-normalized. Numeric representations such as `1` and
`1.0` may differ. This is Python JSON canonicalization, not RFC 8785.

The selected JSON envelope is ASCII-escaped, key-sorted and SHA-256 hashed.
Its serialized size is limited to `max_prefix_bytes` (default 1 MiB). Structure
validation also limits nesting and node count. Invalid observations do not
change history. This diagnostic may raise; the surrounding application decides
how to handle invalid inputs without logging their contents.

## State, ordering and privacy

The tracker retains only the identity tuple and last digest. Its LRU capacity is
bounded by `max_tracked`; an evicted pair or a pair observed after `clear()` or a
process restart returns `new`. Retained state is bounded, but request parsing and
canonicalization still use temporary memory; callers should bound request size
and concurrency. Callers must not mutate input objects during an observation.

Updates and eviction are protected by a lock. Concurrent observations are
compared in the order they reach that lock, which need not be network arrival
order. The utility does not infer causal turn ordering.

No raw prompt or tool-schema content is retained or logged by the tracker.
Identifiers are retained: use opaque IDs, not credentials or personal data.
Digests reveal equality and may permit guessing low-entropy input. They are
not a privacy boundary and should not automatically be exported or logged.

## What it does not establish

This is a fingerprint of selected JSON fields, not token-identical model input.
It does not inspect chat templates, tokenization, provider prefix selection,
cache keys, cached tokens, KV-cache contents, request scheduling or timings.
Neither a changed nor a stable result establishes any cache hit/miss, cache
flush, reprocessing, latency improvement, model behavior or performance result.
