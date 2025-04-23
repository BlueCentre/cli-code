# chuk_mcp/chuk_mcp.mcp_client/messages/initialize/mcp_server_capabilties.py
from typing import Optional

from mcp_code.mcp_client.mcp_pydantic_base import Field, McpPydanticBase


class MCPServerCapabilities(McpPydanticBase):
    logging: dict = Field(default_factory=dict)
    prompts: Optional[dict] = None
    resources: Optional[dict] = None
    tools: Optional[dict] = None
    experimental: Optional[dict] = None
