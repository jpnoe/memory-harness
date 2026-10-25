"""Offline synthetic retrieval demonstration; no model download or API call."""
import json
from pathlib import Path
import tempfile

import numpy as np

from memory import SemanticMemory


class KeywordEncoder:
    """Toy deterministic features for inspecting mechanics, not semantic quality."""
    vocabulary = ("sensor", "midi", "database", "sleep")

    def encode(self, text):
        lowered = text.lower()
        return np.array([float(word in lowered) for word in self.vocabulary] + [0.1])


def run_demo():
    with tempfile.TemporaryDirectory() as directory:
        messages = [
            {"id": "sensor-notes", "role": "user", "content": "Sensor pulses become MIDI control values. 🌸"},
            {"id": "sensor-response", "role": "assistant", "content": "Check sensor timing before smoothing MIDI output. 🌸"},
            {"id": "storage-notes", "role": "user", "content": "The database keeps a local observation history. 💡"},
            {"id": "recent-note", "role": "assistant", "content": "Keep the sleep logging example separate. 💡"},
        ]
        memory = SemanticMemory(str(Path(directory) / "chat.json"), str(Path(directory) / "emojis.json"), messages=messages, encoder=KeywordEncoder())
        query = {"id": "synthetic-query", "role": "user", "content": "Review the sensor MIDI controller. 🌸"}
        before = memory.link_graph.get_links("sensor-notes").get("sensor-response", 0)
        retrieved, recent = memory.get_context(query, top_k=2)
        after = memory.link_graph.get_links("sensor-notes").get("sensor-response", 0)
        return {
            "mode": "offline synthetic fixture with a toy keyword encoder",
            "query": query["content"],
            "retrieved": [{"id": m["id"], "content": m["content"]} for m in retrieved],
            "recent_ids": [m["id"] for m in recent],
            "co_retrieval_edge": {"from": "sensor-notes", "to": "sensor-response", "before": before, "after": after},
            "interpretation": "Demonstrates selection and reinforcement mechanics; not an embedding-quality or emotional-validity evaluation.",
        }


def main():
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
