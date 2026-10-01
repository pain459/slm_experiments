import pytest
from src.distill.schema import parse_strict_json_object, validate_teacher_object


def test_strict_json_accepts_exact_object():
    obj = parse_strict_json_object('{"a": 1}')
    assert obj == {"a": 1}


def test_strict_json_rejects_markdown_fence():
    with pytest.raises(ValueError):
        parse_strict_json_object('```json\n{"a": 1}\n```')


def test_teacher_schema_requires_all_fields():
    errors = validate_teacher_object({"problem": "x"})
    assert errors
    assert any("missing_fields" in x for x in errors)
