from __future__ import annotations

from collections import deque

from app.engine.matching.models import GeometryResult


def build_connectivity_graph(image_count: int,
                             verified_pairs: list[tuple[int, int, GeometryResult]]) -> dict:
    if image_count < 0:
        raise ValueError("image_count cannot be negative")
    adjacency = {i: set() for i in range(image_count)}
    edges = []
    for left, right, result in verified_pairs:
        if not (0 <= left < image_count and 0 <= right < image_count):
            raise ValueError("Pair index is outside image_count")
        if left == right:
            raise ValueError("Self-pairs are not valid graph edges")
        if result.verified:
            adjacency[left].add(right)
            adjacency[right].add(left)
            edges.append({"image_indices": [left, right], **result.to_dict()})
    components, unseen = [], set(adjacency)
    while unseen:
        queue = deque([min(unseen)])
        unseen.remove(queue[0])
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
        components.append(sorted(component))
    return {
        "image_count": image_count,
        "verified_edge_count": len(edges),
        "components": components,
        "isolated_image_indices": sorted(i for i, neighbours in adjacency.items() if not neighbours),
        "edges": edges,
        "interpretation": "Pairwise geometric connectivity is not proof of complete 3D surface coverage.",
    }
