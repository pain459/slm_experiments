from src.distill.curriculum import iter_cells, load_curriculum


def test_python_curriculum_is_ordered_and_comprehensive():
    c = load_curriculum("01_distill/curriculum/python.yaml")
    cells = list(iter_cells(c, ["implementation"]))
    topic_ids = []
    for cell in cells:
        if cell.topic_id not in topic_ids:
            topic_ids.append(cell.topic_id)
    assert topic_ids[0] == "P00"
    assert topic_ids[-1] == "P62"
    assert len(topic_ids) == 63
    assert all(0 <= x.level <= 5 for x in cells)


def test_dsa_curriculum_is_ordered_and_comprehensive():
    c = load_curriculum("01_distill/curriculum/dsa.yaml")
    cells = list(iter_cells(c, ["implementation"]))
    topic_ids = []
    for cell in cells:
        if cell.topic_id not in topic_ids:
            topic_ids.append(cell.topic_id)
    assert topic_ids[0] == "D00"
    assert topic_ids[-1] == "D61"
    assert len(topic_ids) == 62
    olympiad = [x for x in cells if x.topic_id == "D61"]
    assert {x.level for x in olympiad} == {4, 5}
