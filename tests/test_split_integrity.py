def test_placeholder_integrity_rule():
    # Real train/eval IDs are checked by scripts/check_integrity.py once datasets exist.
    assert set(["a","b"]).isdisjoint({"c"})
