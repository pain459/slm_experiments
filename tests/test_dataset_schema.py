from src.common.schema import validate_record

def test_valid_record():
    x={"id":"1","source":"x","language":"python","category":"python","skill":"loops","difficulty":1,"task_type":"implementation","prompt":"Write code"}
    assert validate_record(x)==[]

def test_missing_prompt():
    x={"id":"1","source":"x","language":"python","category":"python","skill":"loops","difficulty":1,"task_type":"implementation"}
    assert validate_record(x)
