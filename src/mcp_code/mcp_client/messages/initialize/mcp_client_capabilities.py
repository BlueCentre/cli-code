# chuk_mcp/chuk_mcp.mcp_client/messages/initialize/chuk_mcp.mcp_client_capabilties.py
from mcp_code.mcp_client.mcp_pydantic_base import Field, McpPydanticBase


class MCPClientCapabilities(McpPydanticBase):
    roots: dict = Field(default_factory=lambda: {"listChanged": True})
    sampling: dict = Field(default_factory=dict)
    experimental: dict = Field(default_factory=dict)
