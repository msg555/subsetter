from subsetter.solver import (
    CycleException,
    order_graph,
    reverse_graph,
    toposort_forward,
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
            assert all(order[u] < order[v] for v in edges)


def test_toposort_forward():
    G = {
        "a": set("bcd"),
        "b": set(),
        "e": set(),
        "c": set("de"),
        "d": set("c"),
    }
    try:
        toposort_forward(G, "a")
    except CycleException as exc:
        assert_cyclic_shift(exc.cycle, ["c", "d"])
    else:
        assert False

    G["c"].remove("d")
    assert_valid_sort(G, set("abcde"), toposort_forward(G, "a"))


def test_toposort_forward_deep():
    N = 10000
    G = {i: {(i + 1) % N} for i in range(N)}

    try:
        toposort_forward(G, 0)
    except CycleException as exc:
        assert_cyclic_shift(exc.cycle, list(range(N)))
    else:
        assert False

    G[N - 1].remove(0)
    assert_valid_sort(G, set(range(N)), toposort_forward(G, 0))


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
