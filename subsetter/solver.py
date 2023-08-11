import copy
import json
from typing import Dict, List, Optional, Set, TypeVar

NodeT = TypeVar("NodeT")
GraphT = Dict[NodeT, List[NodeT]]


class InversionException(Exception):
    pass


def dfs(
    G: GraphT,
    u: NodeT,
    *,
    visited: Optional[Dict[NodeT, int]] = None,
    avoid: Optional[Set[NodeT]] = None,
    color: int = 0,
) -> int:
    if visited is None:
        visited = {}
    if u in visited or (avoid and u in avoid):
        return 0
    visited[u] = color

    result = 1
    for v in G[u]:
        result += dfs(G, v, visited=visited, avoid=avoid, color=color)
    return result


def subgraph(G: GraphT, nodes: Set[NodeT], *, invert: bool = False) -> GraphT:
    G_sub = {}
    for u, edges in G.items():
        if (u in nodes) != invert:
            G_sub[u] = [v for v in edges if (v in nodes) != invert]
    return G_sub


def reverse_graph(G: GraphT, *, union=False) -> GraphT:
    RG = {u: (list(edges) if union else []) for u, edges in G.items()}
    for u, edges in G.items():
        for v in edges:
            RG[v].append(u)
    return RG


def invert_edges_into(G: GraphT, u: NodeT) -> GraphT:
    G = copy.deepcopy(G)
    RG = reverse_graph(G)
    parent: Dict[NodeT, Optional[NodeT]] = {}

    def invert_dfs(u: NodeT, *, p: Optional[NodeT] = None) -> None:
        if u in parent:
            path_u = []
            u_link: Optional[NodeT] = u
            while u_link is not None:
                path_u.append(u_link)
                u_link = parent[u_link]

            path_p = [u]
            u_link = p
            while u_link is not None:
                path_p.append(u_link)
                u_link = parent[u_link]

            while len(path_u) > 1 and path_u[-2:] == path_p[-2:]:
                path_p.pop()
                path_u.pop()

            raise InversionException(
                f"Inverting edges creates cycle: {'->'.join(str(u) for u in path_u)} and {'->'.join(str(u) for u in path_p)}"
            )
        parent[u] = p
        G[u] = [v for v in G[u] if v not in parent]
        G[u].extend(RG[u])
        for v in RG[u]:
            invert_dfs(v, p=u)

    invert_dfs(u)
    return G


def order_graph(G: Dict[NodeT, List[NodeT]], source: NodeT) -> List[NodeT]:
    # Proposed algorithm
    #   If only super-source remains we're done
    #
    #   Find any sink node u
    #      If deleting u does not disconnect the undirected graph, delete u and solve recursively.
    #      Place u last.
    #
    #      Otherwise recursively solve on graph reachable from super source. Place u. Reverse edges
    #      up from u, bail if a cycle found. Solve on remaining graph treating u as new
    #      super-source.

    # Base case, graph only contains the source node which doesn't get placed.
    if len(G) <= 1:
        return []

    G_uni = reverse_graph(G, union=True)

    # Look for a simple pivot first. Sink that does not disconnect.
    for u, edges in G.items():
        if edges:
            continue
        assert u is not source

        # Check if removing u disconnects G, if not pivot on it.
        amt = dfs(G_uni, source, avoid={u})
        if amt + 1 == len(G):
            order = order_graph(subgraph(G, {u}, invert=True), source)
            order.append(u)
            return order

    # Otherwise we know we'll need to pivot on a node that disconnects the graph.
    options = set()
    for u, edges in G.items():
        if edges:
            continue
        assert u is not source

        options.add(u)

        # Find all nodes still reachable from source
        visited: Dict[NodeT, int] = {}
        dfs(G_uni, source, visited=visited, avoid={u})

        # Find remaining subgraph and attempt to invert edges
        G_rem = subgraph(G, set(visited), invert=True)

        try:
            G_rem = invert_edges_into(G_rem, u)
        except InversionException as exc:
            print(f"Attempted to pivot on {u} but failed with:", exc)
        else:
            order = order_graph(subgraph(G, set(visited)), source)
            order.append(u)
            order.extend(order_graph(G_rem, u))
            return order

    raise ValueError(f"Could not find sink to pivot on, options {options}")


def main():
    with open("graph.json", "r", encoding="utf-8") as fgraph:
        graph = json.load(fgraph)

    visited = {}
    dfs(reverse_graph(graph, union=True), "", visited=visited)
    graph = subgraph(graph, set(visited))

    print(order_graph(graph, ""))


if __name__ == "__main__":
    main()
