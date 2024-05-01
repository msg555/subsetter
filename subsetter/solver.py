import logging
from typing import Dict, Iterable, Iterator, List, Set, Tuple, TypeVar

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


def toposort_reachable(
    G: GraphT,
    sources: Iterable[NodeT],
    ignore: Set[NodeT],
) -> List[NodeT]:
    """
    Returns a topological sort of all nodes strongly reachable from `sources`
    without travelling through any nodes in the `ignore` set except for the
    starting element from `sources`.

    All elements of `sources` must be in the `ignore` set.
    """

    weight: Dict[NodeT, int] = {}

    for source_u in sources:
        assert source_u in ignore

        for u in G[source_u]:
            if u in ignore or u in weight:
                continue

            on_stk: Set[NodeT] = {u}
            stk: List[Tuple[NodeT, Iterator[NodeT]]] = [(u, iter(G[u]))]
            weight[u] = 1

            while stk:
                u, it = stk[-1]
                try:
                    v = next(it)
                except StopIteration:
                    stk.pop()
                    on_stk.remove(u)
                    if stk:
                        weight[stk[-1][0]] += weight[u]
                    continue

                if v in ignore:
                    continue
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
    - For every other `u`
      - let `f_u` be # of `v` such that `v` -> `u` in G and `v` is ordered before `u`
      - let `b_u` be # of `v` such that `u` -> `v` in G and `v` is ordered before `u`
      - Then *exactly* one of `f_u` or `b_u` will be 0
      - This ensures `u` can be sampled by taking either the union of several forward
        relationships `v`->`u` or the intersection of several backwards relationships
        `u`->`v`.

    Raises CycleException if no solution can be found.
    """
    # Algorithm approach:
    #
    #   Let G_0 = G if `source` has outgoing edges, otherwise rev(G)
    #   Let H_0 = [source]
    #
    #   Then recursively define
    #     G_{i+1} = rev(G_i)
    #     H_{i+1} = toposort_reachable(G_{i+1}, H_i, union(H_j for j < i))
    #
    #   Then final ordering is concatenation of all H_i

    RG = reverse_graph(G)

    if not G[source]:
        G, RG = RG, G

    vis = {source}
    H = [source]
    result = [source]
    while H:
        H = toposort_reachable(G, H, vis)
        result.extend(H)
        vis.update(H)
        G, RG = RG, G

    return result
