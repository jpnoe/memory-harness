"""Reproducible ingestion benchmark using synthetic conversational data."""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from memory import SemanticMemory


class CheapEncoder:
    """Fast deterministic encoder that exposes Python-side ingestion costs."""

    def __init__(self):
        self.calls = 0
        self.texts_encoded = 0

    @staticmethod
    def _vector(text):
        value = (sum(text.encode("utf-8")) % 997) / 997
        return np.array([1.0, value, value**2, 1.0 - value], dtype=float)

    def encode(self, texts):
        self.calls += 1

        if isinstance(texts, str):
            self.texts_encoded += 1
            return self._vector(texts)

        texts = list(texts)
        self.texts_encoded += len(texts)
        return np.stack([self._vector(text) for text in texts])


def build_messages(count):
    topics = ("memory", "music", "mood", "software", "neuroscience")

    return [
        {
            "id": f"message-{index}",
            "role": "user" if index % 2 == 0 else "assistant",
            "content": (
                f"Synthetic message {index} about {topics[index % len(topics)]}. "
                f"{'Additional context. ' * (index % 6)}"
                f"{'🌸' if index % 10 == 0 else ''}"
            ),
        }
        for index in range(count)
    ]


def benchmark(count):
    messages = build_messages(count)
    encoder = CheapEncoder()

    with tempfile.TemporaryDirectory() as directory:
        start = time.perf_counter()

        memory = SemanticMemory(
            str(Path(directory) / "chat.json"),
            str(Path(directory) / "emojis.json"),
            messages=messages,
            encoder=encoder,
        )

        elapsed = time.perf_counter() - start

    assert len(memory.messages) == count

    return elapsed, encoder.calls, encoder.texts_encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 2500, 5000],
    )
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    print("messages,seconds,messages_per_second,encoder_calls,texts_encoded")

    for count in args.sizes:
        results = [benchmark(count) for _ in range(args.repeats)]
        elapsed = statistics.median(result[0] for result in results)
        calls = results[0][1]
        texts = results[0][2]

        print(f"{count},{elapsed:.6f},{count / elapsed:.2f},{calls},{texts}")


if __name__ == "__main__":
    main()