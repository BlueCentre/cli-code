# chuk_mcp/chuk_mcp.mcp_client/messages/initialize/mcp_server_info.py
from mcp_code.mcp_client.mcp_pydantic_base import Field, McpPydanticBase


class MCPServerInfo(McpPydanticBase):
    name: str
    version: str
