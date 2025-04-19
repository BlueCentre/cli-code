# tests/mcp_tests/mcp_client/test_mcp_pydantic_base_real.py
import json
from typing import Any, Dict, List, Optional

import pytest

# Import the base and also the core exception if available
from mcp_code.mcp_client.mcp_pydantic_base import ConfigDict, Field, McpPydanticBase
from mcp_code.mcp_client.mcp_pydantic_base import ValidationError as McpValidationError

try:
    from pydantic_core import ValidationError as PydanticCoreValidationError

    EffectiveValidationError = PydanticCoreValidationError
except ImportError:
    # Fallback if pydantic_core isn't directly importable or if fallback is active
    EffectiveValidationError = McpValidationError

# Skip if Pydantic is not actually available (means fallback is active)
if McpPydanticBase.__module__ != "pydantic.main":
    pytest.skip("Skipping real Pydantic tests, fallback is active", allow_module_level=True)

# --- Test Classes using Real Pydantic ---


class RealSimpleModel(McpPydanticBase):
    model_config = ConfigDict(extra="allow")  # Allow extra fields like Pydantic v2

    name: str
    value: Optional[int] = None
    items: List[str] = Field(default_factory=list)


class RealNestedModel(McpPydanticBase):
    model_config = ConfigDict(extra="allow")

    id: int
    simple: RealSimpleModel
    config: Dict[str, Any] = Field(default_factory=dict)


# --- Test Cases ---


def test_real_pydantic_simple_instantiation():
    """Test basic instantiation with real Pydantic."""
    m = RealSimpleModel(name="test")
    assert m.name == "test"
    assert m.value is None
    assert m.items == []

    m2 = RealSimpleModel(name="test2", value=123, items=["a", "b"])
    assert m2.name == "test2"
    assert m2.value == 123
    assert m2.items == ["a", "b"]


def test_real_pydantic_missing_required_field():
    """Test real Pydantic ValidationError for missing fields."""
    # Explicitly catch the expected core validation error
    with pytest.raises(EffectiveValidationError):
        RealSimpleModel(value=10)


def test_real_pydantic_default_factory():
    """Test default_factory with real Pydantic."""
    m1 = RealSimpleModel(name="m1")
    m2 = RealSimpleModel(name="m2")
    assert m1.items == []
    assert m2.items == []
    m1.items.append("item1")  # type: ignore
    assert m1.items == ["item1"]
    assert m2.items == []


def test_real_pydantic_nested_instantiation():
    """Test nested instantiation with real Pydantic."""
    nested_data = {
        "id": 1,
        "simple": {"name": "nested_simple", "value": 456, "items": ["x"]},
        "config": {"setting": True},
    }
    nm = RealNestedModel(**nested_data)
    assert nm.id == 1
    assert isinstance(nm.simple, RealSimpleModel)
    assert nm.simple.name == "nested_simple"
    assert nm.simple.value == 456
    assert nm.simple.items == ["x"]
    assert nm.config == {"setting": True}


def test_real_pydantic_model_dump():
    """Test model_dump with real Pydantic."""
    m = RealSimpleModel(name="dump_test", value=100, items=["c"])
    dumped = m.model_dump()
    expected = {"name": "dump_test", "value": 100, "items": ["c"]}
    assert dumped == expected


def test_real_pydantic_model_dump_nested():
    """Test nested model_dump with real Pydantic."""
    nested_data = {
        "id": 3,
        "simple": {"name": "nested_dump", "value": 789, "items": ["y", "z"]},
        "config": {"flag": False},
    }
    nm = RealNestedModel(**nested_data)
    dumped = nm.model_dump()
    expected = {
        "id": 3,
        "simple": {"name": "nested_dump", "value": 789, "items": ["y", "z"]},
        "config": {"flag": False},
    }
    assert dumped == expected


def test_real_pydantic_model_dump_exclude_none():
    """Test model_dump with exclude_none=True with real Pydantic."""
    m = RealSimpleModel(name="exclude_none_test")
    dumped = m.model_dump(exclude_none=True)
    expected = {"name": "exclude_none_test", "items": []}
    assert dumped == expected


def test_real_pydantic_model_dump_exclude_set():
    """Test model_dump with exclude as a set with real Pydantic."""
    m = RealSimpleModel(name="exclude_set_test", value=200, items=["d"])
    dumped = m.model_dump(exclude={"value", "items"})
    expected = {"name": "exclude_set_test"}
    assert dumped == expected


def test_real_pydantic_model_dump_json():
    """Test model_dump_json with real Pydantic."""
    m = RealSimpleModel(name="json_test", value=400, items=["f"])
    json_str = m.model_dump_json()
    expected_dict = {"name": "json_test", "value": 400, "items": ["f"]}
    # Real Pydantic defaults to compact JSON
    assert json_str == '{"name":"json_test","value":400,"items":["f"]}'
    assert json.loads(json_str) == expected_dict


def test_real_pydantic_model_dump_json_indent():
    """Test model_dump_json with indentation with real Pydantic."""
    m = RealSimpleModel(name="indent_test", value=500)
    json_str = m.model_dump_json(indent=2, exclude_none=True)
    expected_dict = {"name": "indent_test", "value": 500, "items": []}
    # Pydantic v2 uses \n, not \r\n
    expected_json = '{\n  "name": "indent_test",\n  "value": 500,\n  "items": []\n}'
    # Handle potential minor whitespace differences in output if necessary
    # For example, by loading and comparing dicts:
    # assert json.loads(json_str) == expected_dict
    assert json_str.strip() == expected_json.strip()
    assert json.loads(json_str) == expected_dict  # Double check loaded dict


def test_real_pydantic_extra_fields():
    """Test handling of extra fields with real Pydantic (requires ConfigDict)."""
    m = RealSimpleModel(name="extra", extra_field="allowed")
    assert m.name == "extra"
    assert m.extra_field == "allowed"
    dumped = m.model_dump()
    assert dumped["extra_field"] == "allowed"
