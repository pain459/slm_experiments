from calculator import divide

def test_normal():
    assert divide(6, 2) == 3

def test_zero():
    assert divide(1, 0) is None
