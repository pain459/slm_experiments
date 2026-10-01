from src.agent.horizon import classify_horizon

def test_horizons():
    assert classify_horizon(2)=='H1'; assert classify_horizon(5)=='H2'; assert classify_horizon(15)=='H3'; assert classify_horizon(30)=='H4'
