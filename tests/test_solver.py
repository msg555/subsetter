import random

import pytest

from subsetter.solver import (
    CycleException,
    order_graph,
    reverse_graph,
    toposort,
    toposort_reachable,
)


def assert_cyclic_shift(a, b):
    assert len(a) == len(b)
    assert any(
        all(a[i] == b[(i + shift) % len(b)] for i in range(len(b)))
        for shift in range(len(b))
    )


def assert_valid_sort(G, nodes, sort):
    assert len(nodes) == len(sort)
    assert nodes == set(sort)
    order = {u: i for i, u in enumerate(sort)}
    for u, edges in G.items():
        if u in nodes:
            assert all(order[u] < order[v] for v in edges if v in nodes)


def test_toposort_reachable():
    G = {
        "a": set("bcd"),
        "b": set(),
        "e": set(),
        "c": set("de"),
        "d": set("c"),
    }
    try:
        toposort_reachable(G, ["a"], set("a"))
    except CycleException as exc:
        assert_cyclic_shift(exc.cycle, ["c", "d"])
    else:
        assert False

    G["c"].remove("d")
    assert_valid_sort(G, set("bcde"), toposort_reachable(G, ["a"], set("a")))
    assert_valid_sort(G, set("bce"), toposort_reachable(G, ["a"], set("ad")))


def test_toposort_forward_deep():
    N = 10000
    G = {i: {(i + 1) % N} for i in range(N)}
    G[-1] = {0}

    try:
        toposort_reachable(G, [-1], {-1})
    except CycleException as exc:
        assert_cyclic_shift(exc.cycle, list(range(N)))
    else:
        assert False

    assert_valid_sort(G, set(range(1, N)), toposort_reachable(G, [0], {0}))


def test_reverse_graph():
    G = {
        "a": set("bcd"),
        "b": set(),
        "c": set("de"),
        "d": set("c"),
        "e": set(),
    }
    assert reverse_graph(G) == {
        "a": set(),
        "b": set("a"),
        "c": set("ad"),
        "d": set("ac"),
        "e": set("c"),
    }
    assert reverse_graph(G, union=True) == {
        "a": set("bcd"),
        "b": set("a"),
        "c": set("ade"),
        "d": set("ac"),
        "e": set("c"),
    }


def test_order_graph():
    G = {
        "a": set("b"),
        "b": set(),
        "c": set("be"),
        "d": set("be"),
        "e": set(),
        "f": set("e"),
        "g": set("ef"),
    }

    # Verify yields one of the valid solutions
    order = order_graph(G, "a")
    assert order in (list("abcdefg"), list("abdcefg"))


def test_order_graph_deep():
    N = 2 * 10000 + 1
    G = {i: set() for i in range(N)}
    for i in range(N // 2):
        G[2 * i + 1].add(2 * i)
        G[2 * i + 1].add(2 * i + 2)
    assert order_graph(G, 0) == list(range(N))


def _validate_ordering(G, source, order) -> None:
    RG = reverse_graph(G)
    reachable = {source}
    q = [source]
    for u in q:
        for v in G[u] | RG[u]:
            if v not in reachable:
                reachable.add(v)
                q.append(v)  # pylint: disable=modified-iterating-list

    assert len(order) == len(reachable)
    assert set(order) == set(reachable)
    assert order[0] == source

    visited = set()  # type: ignore
    for u in order:
        cnt_fwd = sum(1 for v in G[u] if v in visited)
        cnt_rev = sum(1 for v in RG[u] if v in visited)
        assert cnt_fwd == 0 or cnt_rev == 0
        assert u == source or cnt_fwd or cnt_rev
        visited.add(u)


def test_order_real():
    G = {
        "public.actor": set(),
        "public.store": {"public.staff", "public.address"},
        "public.address": {"public.city"},
        "public.city": set(),
        "public.staff": {"public.address"},
        "public.category": set(),
        "public.customer": {"public.address"},
        "public.film_actor": {"public.film", "public.actor"},
        "public.film": {"public.language"},
        "public.language": set(),
        "public.film_category": {"public.film", "public.category"},
        "public.inventory": {"public.film"},
        "public.rental": {"public.staff", "public.customer", "public.inventory"},
        "public.payment": {"public.rental", "public.staff", "public.customer"},
        "": {"public.rental"},
    }
    _validate_ordering(G, "", order_graph(G, ""))


def test_order_two_phase():
    G = {
        "a": set("de"),
        "b": set("cde"),
        "c": set("e"),
        "d": set("e"),
        "e": set(),
    }
    _validate_ordering(G, "a", order_graph(G, "a"))


def test_order_fuzz():
    ITERS = 1000
    rng = random.Random(555)
    for _ in range(ITERS):
        edge_p = rng.random()
        N = rng.randrange(2, 100)
        G = {u: {v for v in range(u + 1, N) if rng.random() < edge_p} for u in range(N)}
        _validate_ordering(G, 0, order_graph(G, 0))


def test_toposort():
    G = {
        "a": set("bcd"),
        "b": set(),
        "e": set(),
        "c": set("e"),
        "d": set("c"),
    }
    order = toposort(G)
    assert_valid_sort(reverse_graph(G), set(G), order)

    G = {
        "a": set("bcde"),
        "b": set("cde"),
        "c": set("de"),
        "d": set("e"),
        "e": set(),
    }
    order = toposort(G)
    assert order == list("edcba")

    G = {
        "a": set("bcde"),
        "b": set("cde"),
        "c": set("de"),
        "d": set("e"),
        "e": set("a"),
    }
    with pytest.raises(ValueError):
        toposort(G)
