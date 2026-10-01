from src.distill.dedup import StreamingDeduper


def test_exact_prompt_duplicate_rejected():
    d = StreamingDeduper(near_threshold=None)
    first, _ = d.check("a", "Add two numbers", "def f(): pass", "P00")
    second, _ = d.check("b", "  ADD   TWO numbers ", "different", "P00")
    assert not first.duplicate
    assert second.duplicate
    assert second.matched_id == "a"


def test_near_duplicate_same_bucket_rejected():
    d = StreamingDeduper(near_threshold=0.75)
    d.check("a", "Implement binary search over a sorted integer array", "x", "D06")
    decision, _ = d.check("b", "Implement binary search on a sorted integer array", "y", "D06")
    assert decision.duplicate
    assert decision.reason == "near_prompt"
