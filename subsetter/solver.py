import logging
from typing import Dict, Iterator, List, Set, Tuple, TypeVar

NodeT = TypeVar("NodeT")
GraphT = Dict[NodeT, Set[NodeT]]


LOGGER = logging.getLogger(__name__)


class SolverException(ValueError):
    """
    Raised if no solution can be found when ordering a graph.
    """


class CycleException(SolverException):
    def __init__(self, cycle: List[NodeT]) -> None:
        super().__init__(f"Cannot solve due to cycle {cycle}")
        self.cycle = cycle


def toposort(G: GraphT) -> List[NodeT]:
    """
    Returns a topological sort such that each node will appear in the ordering
    after all nodes reachable from it.

    Raises ValueError if there are cycles in the graph.
    """

    q = []
    for u, edges in G.items():
        if not edges:
            q.append(u)

    RG = reverse_graph(G)
    deg = {u: len(edges) for u, edges in G.items()}
    for u in q:
        for v in RG[u]:
            deg[v] -= 1
            if not deg[v]:
                # This is intended and well-defined behavior
                # pylint: disable=modified-iterating-list
                q.append(v)

    if len(q) < len(G):
        raise ValueError("Cycle detected in graph")

    return q


def toposort_forward(
    G: GraphT,
    u: NodeT,
) -> List[NodeT]:
    """
    Returns either a topological sort of nodes reachable through forward edges
    from `u` or raises a CycleException if a cycle is detected.
    """

    weight: Dict[NodeT, int] = {u: 1}
    on_stk: Set[NodeT] = {u}
    stk: List[Tuple[NodeT, Iterator[NodeT]]] = [(u, iter(G[u]))]

    while stk:
        u, it = stk[-1]
        try:
            v = next(it)
            if v in on_stk:
                cyc = []
                for w, _ in reversed(stk):
                    cyc.append(w)
                    if w == v:
                        break
                raise CycleException(cyc[::-1])
            if v in weight:
                weight[u] += weight[v]
            else:
                weight[v] = 1
                on_stk.add(v)
                stk.append((v, iter(G[v])))
        except StopIteration:
            stk.pop()
            on_stk.remove(u)
            if stk:
                weight[stk[-1][0]] += weight[u]
            continue

    return [
        x[0] for x in sorted(weight.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ]


def reverse_graph(G: GraphT, *, union=False) -> GraphT:
    """
    Constructs the reverse graph of `G` where edges are oriented in the
    opposite direction.

    If `union` is True it will union of the forward and reverse graph so
    that each edge becomes two edges in the resulting graph.
    """
    RG = {u: (set(edges) if union else set()) for u, edges in G.items()}
    for u, edges in G.items():
        for v in edges:
            RG[v].add(u)
    return RG


def order_graph(G: GraphT, source: NodeT) -> List[NodeT]:
    """
    Will return a ordering of the graph that is suitable for sampling
    given the foreign key dependencies represented in G. Only nodes that are weakly
    reachable from `source` will be included in the ordering.

    The ordering will satisfy the following properties:
    - `source` will be first
    - For every `u`
      - let `f_u` be # of `v` such that `v` -> `u` in G and `v` is ordered before `u`
      - let `b_u` be # of `v` such that `u` -> `v` in G and `v` is ordered before `u`
      - Then either `f_u` = 0 or `b_u` = 0
      - This ensures `u` can be sampled by taking either the union of several forward
        relationships `v`->`u` or the intersection of several backwards relationships
        `u`->`v`.

    Raises CycleException if no solution can be found.
    """
    # Algorithm approach:
    #   Let `H` be the subgraph of nodes strongly reachable from `source`
    #
    #   Begin resulting ordering with any topological sort of `H`
    #
    #   Delete all edges in `H` from `G`
    #
    #   For each node `u` in `H`
    #     Find `S_u` as all nodes strongly reachable in the reversed graph of `G` from `u`
    #     Reverse all edges between two nodes in `S_u` in `G`
    #     Recursively solve rooted at `u` and append ordering, omitting `u` from the start

    # Copy G's graph structure so we can mutate it
    G = {u: set(edges) for u, edges in G.items()}
    RG = reverse_graph(G)

    result = [source]

    q = [source]
    for u in q:
        if not G[u]:
            # Invert all edges reachable from u by only backward edges
            edges: Set[NodeT] = set()
            for v in toposort_forward(RG, u):
                for w in G[v] & edges:
                    G[v].remove(w)
                    RG[v].add(w)
                    G[w].add(v)
                    RG[w].remove(v)
                edges.add(v)

        order = toposort_forward(G, u)
        result.extend(order[1:])

        # Remove all edges between nodes that have been ordered
        nodes = set(order)
        for v in nodes:
            G[v] -= nodes
            RG[v] -= nodes

        # Recursively solve remaining parts of graph that point into the set of nodes
        # we just solved.
        assert not RG[u]
        q.extend(v for v in order[1:] if RG[v])

    return result
