import importlib
import os

import pydantic
import pytest


def test_mcp_pydantic_base_real_pydantic(monkeypatch):
    """
    Test that mcp_pydantic_base uses real Pydantic if available.
    This confirms that we do NOT trigger the fallback logic.
    """
    # Skip this test if fallback is forced.
    if os.environ.get("MCP_FORCE_FALLBACK") == "1":
        pytest.skip("MCP_FORCE_FALLBACK is set; skipping real Pydantic test.")

    import sys

    assert "pydantic" in sys.modules, "Pydantic should be installed for this test."

    monkeypatch.setattr("mcp_code.mcp_client.mcp_pydantic_base.PYDANTIC_AVAILABLE", True)
    import mcp_code.mcp_client.mcp_pydantic_base

    importlib.reload(mcp_code.mcp_client.mcp_pydantic_base)

    # 3) Import the real pydantic version
    from mcp_code.mcp_client.mcp_pydantic_base import ConfigDict, Field, McpPydanticBase

    # Define a test model
    class RealPydanticModel(McpPydanticBase):
        x: int = Field(default=123)
        model_config = ConfigDict(extra="forbid")

    # Check the MRO includes pydantic.BaseModel
    assert pydantic.BaseModel in RealPydanticModel.__mro__, (
        "When Pydantic is installed, McpPydanticBase should be pydantic.BaseModel."
    )

    # Check standard Pydantic behavior
    instance = RealPydanticModel()
    assert instance.model_dump() == {"x": 123}
    instance2 = RealPydanticModel.model_validate({"x": 456})
    assert instance2.x == 456
