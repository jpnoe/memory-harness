import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from graph import LinkGraph
from harness import build_message_payload, create_chat_completion
from memory import SemanticMemory, cosine_similarity, save_messages


class FixtureEncoder:
    """Fixed vectors exercise retrieval mechanics, not real embedding quality."""
    def encode(self, value):
        def encode_one(text):
            return np.array({
                "alpha": [1, 0, 0],
                "alpha again": [1, 0, 0],
                "bravo": [0, 1, 0],
                "charlie": [0, 0, 1],
                "opposite": [-1, 0, 0],
            }.get(text, [1, 1, 1]), dtype=float)

        if isinstance(value, str):
            return encode_one(value)

        return np.stack([encode_one(text) for text in value])

class RecordingEncoder:
    """Records whether texts arrive individually or as a batch."""

    def __init__(self):
        self.calls = []

    def encode(self, value):
        self.calls.append(value)

        if isinstance(value, str):
            return np.array([1.0, 0.0])

        return np.array(
            [[1.0, float(index + 1)] for index, _ in enumerate(value)],
            dtype=float,
        )

def record(id, content):
    return {"id": id, "role": "user", "content": content}


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.history = str(Path(self.temp.name) / "chat.json")
        self.emojis = str(Path(self.temp.name) / "emojis.json")

    def memory(self, messages=None):
        return SemanticMemory(self.history, self.emojis, messages=messages, encoder=FixtureEncoder())

    def test_filtered_vectors_keep_their_original_record_ids(self):
        memory = self.memory([record("a", "alpha"), record("b", "bravo"), record("c", "charlie")])
        result = memory.find_similar(record("q", "charlie"), top_r=0, top_k=1, top_e=0,
                                     max_depth=1, visited_ids={"a"})
        self.assertEqual([m["id"] for m, _ in result], ["c"])

    def test_linked_neighbor_can_expand_beyond_semantic_seeds(self):
        memory = self.memory([record("a", "alpha"), record("b", "bravo")])
        memory.link_graph.link("a", "b", 2)
        result = memory.find_similar(record("q", "alpha"), top_r=0, top_k=1, top_e=0, max_depth=1)
        self.assertEqual({m["id"] for m, _ in result}, {"a", "b"})
        self.assertTrue(all(math.isfinite(weight) for _, weight in result))

    def test_depth_limit_returns_before_dividing_by_zero(self):
        memory = self.memory([record("a", "alpha")])
        self.assertEqual(memory.find_similar(record("q", "alpha"), max_depth=0), [])
        self.assertEqual(memory.find_similar(record("q", "alpha"), max_depth=1, depth=1), [])

    def test_content_budget_never_appends_an_oversized_memory(self):
        memory = self.memory([record("a", "alpha"), record("b", "bravo")])
        result = memory.find_similar(record("q", "alpha"), max_chars=4)
        self.assertLessEqual(sum(len(m["content"]) for m, _ in result), 4)

    def test_cyclic_graph_terminates_with_unique_context(self):
        memory = self.memory([record("a", "alpha"), record("b", "bravo"), record("c", "charlie")])
        for a, b in [("a", "b"), ("b", "c"), ("c", "a")]:
            memory.link_graph.link(a, b)
        result = memory.find_similar(record("q", "alpha"), top_r=0, max_depth=3)
        ids = [m["id"] for m, _ in result]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), 3)

    def test_reinforcement_counts_one_unordered_pair(self):
        memory = self.memory([record("a", "alpha"), record("b", "alpha again")])
        memory.fill_memory_graph(memory.messages)
        self.assertAlmostEqual(memory.link_graph.get_links("a")["b"], 1)
        memory.fill_memory_graph(memory.messages)
        self.assertAlmostEqual(memory.link_graph.get_links("a")["b"], 2)

    def test_symmetric_history_does_not_double_weights_on_reload(self):
        memory = self.memory([record("a", "alpha"), record("b", "alpha again")])
        memory.fill_memory_graph(memory.messages)
        save_messages(memory.messages, self.history)
        restored = self.memory()
        self.assertAlmostEqual(restored.link_graph.get_links("a")["b"], 1)

    def test_negative_cosine_does_not_become_positive_reinforcement(self):
        memory = self.memory([record("a", "alpha"), record("b", "opposite")])
        memory.fill_memory_graph(memory.messages)
        self.assertNotIn("b", memory.link_graph.get_links("a"))

    def test_existing_emoji_pair_is_reinforced_on_each_occurrence(self):
        memory = self.memory([])
        memory.add_memory(record("a", "first 🌸 💡"))
        memory.add_memory(record("b", "second 🌸 💡"))
        self.assertEqual(memory.emoji_graph.get_links("🌸")["💡"], 2)
        restored = self.memory()
        self.assertEqual(restored.emoji_graph.get_links("🌸")["💡"], 2)

    def test_context_does_not_persist_transient_ranking_flags(self):
        source = record("a", "alpha")
        memory = self.memory([source])
        memory.get_context(record("q", "bravo"))
        stored = json.loads(Path(self.history).read_text())
        self.assertTrue(all("recent" not in m and "weight" not in m for m in stored))
        self.assertNotIn("vector", source)

    def test_empty_history_accepts_first_query(self):
        memory = self.memory([])
        self.assertEqual(memory.get_context(record("q", "alpha")), ([], []))
        self.assertEqual(memory.messages[0]["id"], "q")

    def test_embedding_dimensions_are_consistent(self):
        memory = self.memory([record("a", "alpha")])
        bad = record("b", "bravo")
        bad["vector"] = "[1, 2]"
        with self.assertRaises(ValueError):
            memory.add_memory(bad)

    def test_duplicate_ids_are_rejected(self):
        memory = self.memory([record("a", "alpha")])
        with self.assertRaises(ValueError):
            memory.add_memory(record("a", "bravo"))

    def test_zero_vector_similarity_is_finite(self):
        self.assertEqual(cosine_similarity([[0, 0]], [[1, 0]])[0, 0], 0)

    def test_initial_messages_are_encoded_in_one_batch(self):
        encoder = RecordingEncoder()
        messages = [
            record("a", "alpha"),
            record("b", "bravo"),
            record("c", "charlie"),
        ]

        SemanticMemory(
            self.history,
            self.emojis,
            messages=messages,
            encoder=encoder,
        )

        self.assertEqual(
            encoder.calls,
            [["alpha", "bravo", "charlie"]],
        )


class GraphTests(unittest.TestCase):
    def test_traversal_honors_zero_and_total_item_limits(self):
        graph = LinkGraph()
        graph.link("a", "b", 2)
        self.assertEqual(graph.traverse_greedy(["a"], n_items=0), {})
        self.assertEqual(len(graph.traverse_greedy(["a", "b"], n_items=1)), 1)

    def test_graph_rejects_nonfinite_or_negative_weights(self):
        for weight in [float("nan"), float("inf"), -1]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                LinkGraph().link("a", "b", weight)


class AdapterTests(unittest.TestCase):
    def test_archived_data_is_separate_and_bracketed_text_is_preserved(self):
        result = build_message_payload([record("a", "keep [citation]")], [{"role": "user", "content": "question"}])
        self.assertIn("not instructions", result[0]["content"])
        self.assertIn("[citation]", result[0]["content"])
        self.assertEqual(result[-1], {"role": "user", "content": "question"})

    def test_live_adapter_requires_configuration_before_any_transport(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            create_chat_completion([], [], transport=lambda *args: self.fail("Transport must not run"))

    def test_adapter_request_can_be_verified_without_network(self):
        calls = []
        def transport(url, payload, headers, timeout):
            calls.append((url, payload, headers, timeout))
            return {"choices": [{"message": {"content": " synthetic answer "}}]}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-value", "OPENAI_MODEL": "test-model"}):
            answer = create_chat_completion([], [{"role": "user", "content": "hello"}], transport=transport)
        self.assertEqual(answer, "synthetic answer")
        self.assertEqual(calls[0][1]["model"], "test-model")
        self.assertEqual(calls[0][3], 30)


if __name__ == "__main__":
    unittest.main()
