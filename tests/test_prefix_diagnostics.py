import copy
import io
import math
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from threading import Barrier

from prefix_diagnostics import PrefixTracker, prefix_fingerprint


def messages(policy="alpha", turns=1):
    return [{"role": "system", "content": policy}] + [
        {"role": "user", "content": f"question {i}"} for i in range(turns)
    ]


def observe(tracker, policy="alpha", conversation_id="conversation", model="model", **extra):
    return tracker.observe(conversation_id=conversation_id, model=model,
                           messages=messages(policy), **extra)


class PrefixTests(unittest.TestCase):
    def test_new_stable_and_changed_are_distinct(self):
        tracker = PrefixTracker()
        self.assertEqual(observe(tracker).status, "new")
        self.assertEqual(observe(tracker).status, "stable")
        self.assertEqual(observe(tracker, "beta").status, "changed")

    def test_history_changes_do_not_change_the_selected_prefix(self):
        base = messages()
        later = base + [{"role": "assistant", "content": "answer"},
                        {"role": "system", "content": "later instruction"}]
        self.assertEqual(prefix_fingerprint(base), prefix_fingerprint(later))
        self.assertEqual(prefix_fingerprint(base), prefix_fingerprint(messages(turns=8)))

    def test_changed_prefix_is_reported_without_length_heuristics(self):
        tracker = PrefixTracker()
        tracker.observe(conversation_id="a", model="m", messages=messages("alpha", 8))
        report = tracker.observe(conversation_id="a", model="m", messages=messages("beta", 1))
        self.assertEqual(report.status, "changed")

    def test_conversations_models_and_delimiter_strings_are_isolated(self):
        tracker = PrefixTracker()
        pairs = [("a", "m"), ("b", "m"), ("a", "n"), ("a\0b", "c"), ("a", "b\0c")]
        for i, (conversation, model) in enumerate(pairs):
            self.assertEqual(observe(tracker, str(i), conversation, model).status, "new")
        for i, (conversation, model) in enumerate(pairs):
            self.assertEqual(observe(tracker, str(i), conversation, model).status, "stable")

    def test_nested_object_key_order_is_canonical_but_array_order_matters(self):
        first = [{"name": "tool", "schema": {"b": 2, "a": [1, 2]}}]
        second = [{"schema": {"a": [1, 2], "b": 2}, "name": "tool"}]
        self.assertEqual(prefix_fingerprint(messages(), first), prefix_fingerprint(messages(), second))
        second[0]["schema"]["a"].reverse()
        self.assertNotEqual(prefix_fingerprint(messages(), first), prefix_fingerprint(messages(), second))
        self.assertEqual(prefix_fingerprint(messages()), prefix_fingerprint(messages(), []))

    def test_leading_system_order_role_and_exact_unicode_are_significant(self):
        first = [{"role": "system", "content": "alpha"}, {"role": "system", "content": "beta"}]
        self.assertNotEqual(prefix_fingerprint(first), prefix_fingerprint(first[::-1]))
        developer = [{"role": "developer", "content": "alpha"}, *first]
        self.assertEqual(prefix_fingerprint(developer), prefix_fingerprint([]))
        self.assertNotEqual(prefix_fingerprint(messages("\u00e9")), prefix_fingerprint(messages("e\u0301")))

    def test_non_json_values_and_non_finite_numbers_are_rejected(self):
        class NoCoercion:
            def __repr__(self):
                raise AssertionError("Object coercion must not run")
            __str__ = __repr__
        for value in [NoCoercion(), {1, 2}, (1, 2), {1: "value"}, math.nan, math.inf, -math.inf]:
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    prefix_fingerprint(messages(), [value])
        for bad in [None, "messages", [None], [{"content": "alpha"}], [{"role": 3}]]:
            with self.assertRaises((TypeError, ValueError)):
                prefix_fingerprint(bad)

    def test_cycles_and_excessive_depth_are_rejected(self):
        cycle = []; cycle.append(cycle)
        with self.assertRaises(ValueError):
            prefix_fingerprint(messages(), cycle)
        nested = []
        for _ in range(80): nested = [nested]
        with self.assertRaises(ValueError):
            prefix_fingerprint(messages(), nested)

    def test_failed_or_oversized_observation_does_not_update_history(self):
        tracker = PrefixTracker(max_prefix_bytes=512)
        observe(tracker)
        with self.assertRaises(ValueError): observe(tracker, "\u2603" * 100)
        with self.assertRaises(TypeError): observe(tracker, tools={})
        self.assertEqual(observe(tracker).status, "stable")

    def test_configuration_and_identifier_limits(self):
        for bad in [0, -1, True, 1.5, "2"]:
            with self.assertRaises(ValueError): PrefixTracker(max_tracked=bad)
            with self.assertRaises(ValueError): PrefixTracker(max_prefix_bytes=bad)
        for bad in ["", None, "a" * 257, "\u00e9" * 129, "\ud800"]:
            with self.assertRaises(ValueError): observe(PrefixTracker(), conversation_id=bad)
            with self.assertRaises(ValueError): observe(PrefixTracker(), model=bad)

    def test_lru_capacity_refresh_and_clear(self):
        tracker = PrefixTracker(max_tracked=2)
        observe(tracker, conversation_id="a"); observe(tracker, conversation_id="b")
        observe(tracker, conversation_id="a"); observe(tracker, conversation_id="c")
        self.assertEqual(len(tracker), 2)
        self.assertEqual(observe(tracker, conversation_id="a").status, "stable")
        self.assertEqual(observe(tracker, conversation_id="b").status, "new")
        tracker.clear()
        self.assertEqual(len(tracker), 0)
        self.assertEqual(observe(tracker).status, "new")

    def test_concurrent_observations_have_one_first_result_and_stay_bounded(self):
        tracker = PrefixTracker(max_tracked=4)
        barrier = Barrier(12)
        def same(_):
            barrier.wait()
            return observe(tracker).status
        with ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(pool.map(same, range(12)))
            list(pool.map(lambda i: observe(tracker, conversation_id=str(i)), range(80)))
        self.assertEqual(statuses.count("new"), 1)
        self.assertEqual(statuses.count("stable"), 11)
        self.assertLessEqual(len(tracker), 4)

    def test_inputs_are_not_mutated_and_diagnostic_does_not_print(self):
        source = messages(); tools = [{"schema": {"enum": ["a", "b"]}}]
        before = copy.deepcopy((source, tools))
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            PrefixTracker().observe(conversation_id="a", model="m", messages=source, tools=tools)
        self.assertEqual((source, tools), before)
        self.assertEqual(output.getvalue(), "")

    def test_reports_are_immutable_and_contain_only_status_and_digest(self):
        report = observe(PrefixTracker())
        with self.assertRaises(FrozenInstanceError): report.status = "changed"
        self.assertEqual(set(report.__dataclass_fields__), {"status", "fingerprint"})
        self.assertEqual(len(report.fingerprint), 64)


if __name__ == "__main__":
    unittest.main()
