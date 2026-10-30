import numpy as np
from copy import deepcopy
from itertools import combinations
import math
import json
import os
import uuid
import re

from graph import LinkGraph

def cosine_similarity(a, b):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    denominator = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    return np.divide(a @ b.T, denominator, out=np.zeros((len(a), len(b))), where=denominator != 0)


# Basic emoji detection regex
emoji_pattern = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002700-\U000027BF"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "]+", flags=re.UNICODE)

def extract_emojis(text):
    return emoji_pattern.findall(text)

def emoji_density(text):
    emojis = extract_emojis(text)
    if not text:
        return 0
    return len(emojis) / len(text)

def save_emojis(messages, filename):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)

def load_emojis(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.loads(f.read() or "[]") or []
    return []

def save_messages(messages, filename):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)

def load_messages(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.loads(f.read() or "[]") or []
    return []

class SemanticMemory:
    def __init__(self, chat_file, emoji_file, model_name="all-MiniLM-L6-v2", messages=None, *, encoder=None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name)
        self.model = encoder
        self.memories = []  # List of tuples: (text, vector)
        self.messages = []
        self._message_ids = set()
        self.link_graph = LinkGraph()
        self.emojis = load_emojis(emoji_file)
        self.emoji_graph = LinkGraph()
        self.chat_file = chat_file
        self.emoji_file = emoji_file
        messages = deepcopy(messages if messages is not None else load_messages(chat_file))
        # Serialized links are symmetric; restore each undirected edge once.
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                self.add_memory(msg, skip_save=True)
        self._restore_links(self.messages, self.link_graph, "id")
        self._restore_links(self.emojis, self.emoji_graph, "content")

    @staticmethod
    def _restore_links(records, graph, key):
        valid = {record[key] for record in records}
        edges = {}
        for record in records:
            for target, weight in record.get("links", {}).items():
                if target not in valid or target == record[key]:
                    continue
                weight = float(weight)
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("Graph weights must be finite and non-negative")
                pair = tuple(sorted((record[key], target)))
                edges[pair] = max(edges.get(pair, 0.0), weight)
        for (a, b), weight in edges.items():
            graph.link(a, b, weight)

    def add_memory(self, memory, skip_save=False):
        memory = deepcopy(memory)
        memory.pop("recent", None)
        memory.pop("weight", None)
        if "vector" in memory:
            vector = json.loads(memory["vector"])
            vector = np.array(vector)
        else:
            vector = self.model.encode(memory["content"])

        if "id" not in memory:
            memory["id"] = str(uuid.uuid4())
        
        vector = np.asarray(vector, dtype=float)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("An embedding must be a nonempty finite vector")
        if self.memories and vector.shape != self.memories[0][1].shape:
            raise ValueError("Embedding dimensions differ; use one encoder per history")
        if memory["id"] in self._message_ids:
            raise ValueError("Memory IDs must be unique")
        emojis = extract_emojis(memory["content"])
        memory["emojis"] = emojis
        memory["emoji_density"] = emoji_density(memory["content"])
        memory["vector"] = json.dumps(vector.tolist())

        self.memories.append((memory, vector))
        self.messages.append(memory)
        self._message_ids.add(memory["id"])
        if not skip_save:
            save_messages(self.messages, self.chat_file)
            update = False

            for emoji in emojis:
                idx = self.find_emoji_idx(emoji)
                if idx is None:
                    emoji_obj = {"content": emoji, "id": str(uuid.uuid4()), "links": {}}
                    self.emojis.append(emoji_obj)
                    update = True
            
            if emojis:
                self.fill_emoji_graph(emojis)
                save_emojis(self.emojis, self.emoji_file)
        return memory

    def find_related_by_emoji(self, emoji_list, target_ratio, top_k=5, divisor=1.0):
        if not emoji_list:
            return []
        links = self.emoji_graph.traverse_greedy(sorted(emoji_list), n_items=len(emoji_list) + top_k)
        messages = []
        for emoji in links:
            matches = [{**m, "weight": (1.0 if math.isinf(links[emoji]) else links[emoji] / (1.0 + links[emoji])) / divisor} for m in self.messages if emoji in m["emojis"]]
            messages.extend([m for m in matches if "id" in m])

        messages = list({message["id"]: message for message in messages}.values())
        messages.sort(key=lambda x: abs((x.get("emoji_density", 0) or 0) - (target_ratio or 0)))

        # Grab actual memory objects
        return messages[:top_k]
    
    def find_similar(
        self,
        initial_query,
        top_r=2,
        top_k=3,
        top_e=3,
        max_chars=1600,
        visited_ids=None,
        context=None,
        depth=0,
        max_depth=10,
        recents=True
    ):
        def should_exit():
            seen_ids = set()
            total_length = 0
            for mem, _ in context:
                if mem["id"] not in seen_ids:
                    total_length += len(mem["content"])
                    seen_ids.add(mem["id"])
            return total_length >= max_chars or depth >= max_depth
        
        def deduplicate_context():
            context_dict = {}
            for mem, weight in context:
                if mem["id"] not in context_dict or weight > context_dict[mem["id"]][1]:
                    existing = context_dict.get(mem["id"], (None, -1))[0]
                    is_recent = mem.get("recent", False) or (existing and existing.get("recent", False))
                    mem["recent"] = is_recent
                    context_dict[mem["id"]] = (mem, weight)

            ctx = sorted(context_dict.values(), key=lambda x: -x[1])
            return ctx

        if visited_ids is None:
            visited_ids = set()
        if context is None:
            context = []

        if should_exit():
            return deduplicate_context()
        top_r = max(0, top_r)
        top_k = max(0, top_k)
        top_e = max(0, top_e)

        # 1. Get most recent memories
        end_idx = next((i for i, m in enumerate(self.messages) if m["id"] == initial_query["id"]), None) if depth > 0 else len(self.messages)
        if end_idx is None:
            end_idx = len(self.messages)

        recent_memories = [{**m, "weight": 1 / (max_depth - depth)} for m in self.messages[max(end_idx-top_r, 0):end_idx] if m["id"] not in visited_ids]

        query_vec = self.model.encode(initial_query["content"])
        available = [(mem, vec) for mem, vec in self.memories if mem["id"] not in visited_ids]
        vectors = np.array([vec for _, vec in available])
        
        if vectors.size == 0:
            # If no vectors are left to compare, skip similarity calculation
            selected = [] 
        else:
            # Proceed with similarity calculation only if vectors array is not empty
            sims = cosine_similarity([query_vec], vectors)[0]
            ranked = sorted(zip(available, sims), key=lambda x: x[1], reverse=True)
            selected = [memory for ((memory, _), _) in ranked[:top_k] if memory is not None]

        # 2. Traverse memory link graph
        mem_ids = [m["id"] for m in selected]
        linked_ids = self.link_graph.traverse_greedy(mem_ids, n_items=len(mem_ids) + top_k)
        linked_memories = [next(({**m, "weight": (1.0 if math.isinf(linked_ids[lid]) else linked_ids[lid] / (1.0 + linked_ids[lid])) / (max_depth - depth)} for m in self.messages if m["id"] == lid), None) for lid in linked_ids]
        linked_memories = [m for m in linked_memories if m and m["id"] not in visited_ids]

        # 3. Traverse emoji link graph
        emojis = set()
        for mem in recent_memories:
            emojis.update(mem.get("emojis", []))

        emoji_memories = self.find_related_by_emoji(
            emojis, 
            target_ratio=float(np.mean([m.get("emoji_density", 0) for m in recent_memories])) if recent_memories else 0.0,
            top_k=top_e,
            divisor=max_depth - depth
        )
        
        if recents and depth > 0:
            all_memories = recent_memories + linked_memories + emoji_memories
        else:
            all_memories = linked_memories + emoji_memories

        if recents and depth == 0:
            for mem in recent_memories:
                weight = mem["weight"]
                del mem["weight"]
                mem["recent"] = True
                visited_ids.add(mem["id"])
                if sum(len(m["content"]) for m, _ in context) + len(mem["content"]) > max_chars:
                    continue
                context.append((mem, weight))
                if should_exit():
                    return deduplicate_context()

        for mem in all_memories:
            weight = mem["weight"]
            del mem["weight"]
            if mem["id"] in visited_ids:
                continue
            visited_ids.add(mem["id"])
            if sum(len(m["content"]) for m, _ in context) + len(mem["content"]) > max_chars:
                continue
            context.append((mem, weight))
            if should_exit():
                return deduplicate_context()

        if recents:
            for mem in recent_memories:
                # Use the content as the new query
                context = self.find_similar(
                    mem,
                    top_r=top_r - 1,
                    top_k=top_k,
                    top_e=top_e,
                    max_chars=max_chars,
                    visited_ids=visited_ids,
                    context=context,
                    max_depth=max_depth,
                    depth=depth + 1,
                    recents=False
                )
                # Check stop condition after each recursion
                if should_exit():
                    return deduplicate_context()
        
        other_memories = linked_memories + emoji_memories
        for mem in other_memories:
            # Use the content as the new query
            context = self.find_similar(
                mem,
                top_r=top_r,
                top_k=top_k,
                top_e=top_e,
                max_chars=max_chars,
                visited_ids=visited_ids,
                context=context,
                max_depth=max_depth,
                depth=depth + 1
            )
            # Check stop condition after each recursion
            if should_exit():
                return deduplicate_context()

        return deduplicate_context()

    def fill_memory_graph(self, messages):
        by_id = {message["id"]: message for message in messages}
        vectors = {message["id"]: vector for message, vector in self.memories}
        # One reinforcement per unordered pair, using cached embeddings.
        for a, b in combinations(by_id, 2):
            similarity = float(cosine_similarity([vectors[a]], [vectors[b]])[0, 0])
            if similarity > 0:
                self.link_graph.link(a, b, similarity ** 2)
        for message in self.messages:
            message["links"] = dict(self.link_graph.get_links(message["id"]))

    def find_emoji_idx(self, query: str):
        for i, emoji in enumerate(self.emojis):
            if emoji["content"] == query:
                return i
        return None

    def fill_emoji_graph(self, emojis):
        for a, b in combinations(dict.fromkeys(emojis), 2):
            self.emoji_graph.link(a, b)
        for record in self.emojis:
            record["links"] = dict(self.emoji_graph.get_links(record["content"]))

    def remove_trailing_hashtags(self, text):
        words = text.strip().split()
        # Iterate from the end, pop words that start with '#' until a non-hashtag is found
        while words and words[-1].startswith('#'):
            words.pop()
        return ' '.join(words)
    
    def get_context(self, query_obj: dict, top_k=5):
        similar_memories = self.find_similar(query_obj, top_k=top_k)
        similar_memories = [m[0] for m in similar_memories]

        query_obj = self.add_memory(query_obj)

        self.fill_memory_graph([*similar_memories, query_obj])
        
        save_messages(self.messages, self.chat_file)
        stripped_memories = []
        recent_memories = [m for m in similar_memories if m.get("recent", False)]
        for memory in similar_memories:
            if not memory.get("recent", False):
                content = memory["content"]
                content = self.remove_trailing_hashtags(content)
                stripped_memories.append({**memory, "content": content})

        return stripped_memories, recent_memories

    def dump(self):
        return [text for text, _ in self.memories]

