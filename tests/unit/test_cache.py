from earningslora.utils.cache import JsonCache


def test_cache_round_trip(tmp_path):
    c = JsonCache(tmp_path / "cache.db")
    c.set("ns", {"k": 1}, {"value": [1, 2, 3]})
    assert c.get("ns", {"k": 1}) == {"value": [1, 2, 3]}


def test_cache_miss_returns_none(tmp_path):
    c = JsonCache(tmp_path / "cache.db")
    assert c.get("ns", {"missing": True}) is None


def test_cache_namespace_isolation(tmp_path):
    c = JsonCache(tmp_path / "cache.db")
    c.set("ns_a", "shared", {"who": "a"})
    c.set("ns_b", "shared", {"who": "b"})
    assert c.get("ns_a", "shared") == {"who": "a"}
    assert c.get("ns_b", "shared") == {"who": "b"}


def test_cache_overwrite(tmp_path):
    c = JsonCache(tmp_path / "cache.db")
    c.set("ns", "k", {"v": 1})
    c.set("ns", "k", {"v": 2})
    assert c.get("ns", "k") == {"v": 2}


def test_cache_persists_across_instances(tmp_path):
    db = tmp_path / "cache.db"
    JsonCache(db).set("ns", "k", {"v": 1})
    assert JsonCache(db).get("ns", "k") == {"v": 1}


def test_contains(tmp_path):
    c = JsonCache(tmp_path / "cache.db")
    c.set("ns", "k", "value")
    assert ("ns", "k") in c
    assert ("ns", "missing") not in c


def test_payload_order_independence(tmp_path):
    """Hash should be order-independent for dict payloads."""
    c = JsonCache(tmp_path / "cache.db")
    c.set("ns", {"a": 1, "b": 2}, "value")
    assert c.get("ns", {"b": 2, "a": 1}) == "value"
