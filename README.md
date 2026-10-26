# Associative memory harness

An experimental Python system combining pretrained semantic embeddings, weighted memory links, recursive retrieval, recent context, and a separate emoji-association graph. The project was developed by Jacob Patrick Noe with AI collaborators.

The core retrieves archived conversational material and reinforces links between items retrieved together. An optional command-line adapter supplies that context to a chat model. This is an engineering prototype, not an evaluated model of human memory or emotion.

## Run the offline example

Requires Python 3.10 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
python demo.py
python -m unittest discover -s tests -v
```

The example uses a deterministic keyword encoder, a synthetic conversation, and a temporary directory. It makes no model downloads or API requests. It shows retrieved IDs, recent context IDs, and an edge weight before and after reinforcement.

Review verification: all 19 offline tests passed under Python 3.12.14 and NumPy 2.3.5. The full SentenceTransformers model and live API integration were not executed. The API adapter was checked with an injected test transport.

## Use pretrained embeddings

```sh
pip install -e '.[embeddings]'
```

`SemanticMemory` uses `SentenceTransformer("all-MiniLM-L6-v2")` by default. Loading it may download model files. The encoder is injectable so tests and demonstrations can run separately from large model dependencies.

Using SentenceTransformers for pretrained inference is part of this implementation. The project does not demonstrate training or fine-tuning a PyTorch model.

## Optional live chat

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in your shell or secret manager, then run:

```sh
python harness.py --history data/chat.json --emojis data/emojis.json
# Or after installation:
harness-chat --history data/chat.json --emojis data/emojis.json
```

The model ID must be a Chat Completions-compatible model enabled for your account. No live model is selected or contacted by the offline demo. `.env.example` documents variable names; this project does not automatically load dotenv files.

The CLI sends retrieved excerpts and recent conversation to the configured API. Retrieved archive excerpts are labeled as data and kept separate from live conversation turns. This labeling is not a complete prompt-injection defense.

## Retrieval flow

1. Encode a query and rank eligible archived embeddings by cosine similarity.
2. Use selected IDs as seeds for weighted memory-graph traversal. Additional traversal capacity permits expansion beyond those seeds.
3. Retrieve emoji-related memories from recent context using the separate association graph and an emoji-density heuristic.
4. Expand recursively within depth, visited-ID, and retrieved-content character limits.
5. Return archived matches and recent conversation separately.
6. Add the current query and reinforce positive-similarity links among the selected group, once per unordered pair, then persist JSON records.

## Files

| File | Purpose |
| --- | --- |
| `memory.py` | Embeddings, memory storage, retrieval, emoji associations, and reinforcement |
| `graph.py` | Undirected weighted graph and bounded greedy traversal |
| `harness.py` | Optional API adapter and interactive CLI |
| `demo.py` | Synthetic offline example using a toy encoder |
| `tests/test_memory.py` | Regression tests using fixed embeddings and a fake API transport |
| `docs/review.md` | Changes, limits, and evaluation questions |

Runtime history and emoji files belong in ignored `data/`. This source package contains no conversation archive or model weights. Legacy role labels Jake, Lumen, and Lyra remain accepted alongside user and assistant.

## Limitations

- Retrieval uses a full embedding scan and whole-file JSON persistence; no scalable vector database or multi-process transaction model is implemented.
- The inherited depth and channel weighting is heuristic. It is not a calibrated emotional/semantic fusion model.
- The legacy emoji detector groups contiguous matched symbols, uses broad Unicode ranges, and is not a complete grapheme-aware emoji parser. Its matches and density are not validated affect labels.
- Co-retrieval does not prove usefulness. Reinforcement can amplify irrelevant associations.
- Character limits apply to retrieved content, not the full API token budget.
- Stored vectors do not yet record an encoder revision. Changing encoder or dimensions requires a deliberate rebuild of the history.

Next evaluation: compare a frozen semantic-only baseline with graph and emoji channels enabled, measure retrieval relevance on held-out queries, and test corpus/order sensitivity before making effectiveness claims.

## References

- https://sbert.net/docs/package_reference/sentence_transformer/model.html
- https://developers.openai.com/api/reference/overview
- https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
