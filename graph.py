import heapq
import math

class LinkGraph:
    def __init__(self):
        self.graph = {}  # node_id -> { linked_id: weight }

    def link(self, a, b, weight=1.0):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Link weight must be finite and non-negative")
        if a == b:
            return
        self.graph.setdefault(a, {})
        self.graph.setdefault(b, {})
        self.graph[a][b] = self.graph[a].get(b, 0.0) + weight
        self.graph[b][a] = self.graph[b].get(a, 0.0) + weight

    def get_links(self, node):
        return self.graph.get(node, {})

    def traverse_greedy(self, start_ids, n_items=5, min_weight=0.0):
        """Bounded greedy edge-priority expansion; seeds count toward n_items.

        Returned root priorities are infinity, not probability estimates.
        """
        if n_items <= 0:
            return {}
        visited = {}
        heap = []

        for node_id in list(dict.fromkeys(start_ids))[:n_items]:
            visited[node_id] = float('inf')  # Root nodes at top confidence
            for neighbor, weight in self.get_links(node_id).items():
                if weight >= min_weight:
                    # Use negative weight because heapq is a min-heap
                    heapq.heappush(heap, (-weight, neighbor, node_id))

        while heap and len(visited) < n_items:
            neg_weight, node, _ = heapq.heappop(heap)
            if node in visited:
                continue
            visited[node] = -neg_weight  # Restore original weight

            for neighbor, weight in self.get_links(node).items():
                if neighbor not in visited and weight >= min_weight:
                    heapq.heappush(heap, (-weight, neighbor, node))

        return visited  # {node_id: weight}
