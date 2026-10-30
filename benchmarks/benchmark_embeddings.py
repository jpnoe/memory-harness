"""Compare sequential and batched SentenceTransformer inference."""

import argparse
import statistics
import time

import numpy as np
from sentence_transformers import SentenceTransformer


def build_texts(count):
    topics = ("memory", "music", "mood", "software", "neuroscience")

    return [
        (
            f"Message {index} discusses {topics[index % len(topics)]}. "
            f"{'Additional contextual detail. ' * (index % 6)}"
        )
        for index in range(count)
    ]


def measure(operation, repeats):
    durations = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        durations.append(time.perf_counter() - start)

    return statistics.median(durations), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    texts = build_texts(args.count)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=args.device,
    )

    # Warm up model initialization and the selected compute device.
    model.encode(
        texts[:8],
        batch_size=8,
        show_progress_bar=False,
    )

    sequential_seconds, sequential_vectors = measure(
        lambda: np.stack([
            model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for text in texts
        ]),
        args.repeats,
    )

    batch_seconds, batch_vectors = measure(
        lambda: model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ),
        args.repeats,
    )

    maximum_difference = float(
        np.max(np.abs(sequential_vectors - batch_vectors))
    )

    print(f"device: {model.device}")
    print(f"texts: {args.count}")
    print(f"batch size: {args.batch_size}")
    print(f"sequential seconds: {sequential_seconds:.4f}")
    print(f"batch seconds: {batch_seconds:.4f}")
    print(f"speedup: {sequential_seconds / batch_seconds:.2f}x")
    print(f"maximum embedding difference: {maximum_difference:.8f}")


if __name__ == "__main__":
    main()