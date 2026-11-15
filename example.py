"""Synthetic input only; no network requests or model calls."""

from prefix_diagnostics import PrefixTracker

tracker = PrefixTracker(max_tracked=8)
for policy, question in [("Use plain language.", "First question"),
                         ("Use plain language.", "Next question"),
                         ("Use concise language.", "Another question")]:
    report = tracker.observe(
        conversation_id="example-conversation",
        model="example-model",
        messages=[{"role": "system", "content": policy},
                  {"role": "user", "content": question}],
        tools=[{"name": "example_tool", "parameters": {"type": "object"}}],
    )
    print(report.status, report.fingerprint)
