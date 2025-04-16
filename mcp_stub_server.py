import asyncio
import logging

from chuk_mcp import (
    MCPError,
    MCPMessage,
    MCPServerProtocol,
    MCPToolCallRequest,
    MCPToolResult,
    decode_mcp_message,
    encode_mcp_message,
)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 8999


class StubMCPServerProtocol(MCPServerProtocol):
    """A very basic MCP server protocol that just logs and acknowledges messages."""

    async def handle_message(self, message: MCPMessage):
        log.info(f"Stub Server Received: {message.model_dump_json(indent=2)}")

        if message.message_type == "user_message":
            # Send a simple acknowledgement
            response = MCPMessage(
                message_type="assistant_message",
                agent_id=message.agent_id,  # Echo back agent_id
                session_id=message.session_id,  # Echo back session_id
                payload={"text": f"Stub Server received your message: '{message.payload.get('text', '')[:50]}...'"},
            )
            await self.send_message(response)
            log.info(f"Stub Server Sent: {response.model_dump_json(indent=2)}")

        elif message.message_type == "tool_result":
            # Acknowledge tool result
            response = MCPMessage(
                message_type="assistant_message",
                agent_id=message.agent_id,
                session_id=message.session_id,
                payload={"text": f"Stub Server received tool result for {message.payload.get('tool_name', '')}"},
            )
            await self.send_message(response)
            log.info(f"Stub Server Sent: {response.model_dump_json(indent=2)}")

        # Other message types could be handled here (tool_call_request etc.)
        # For the stub, we just log them.


async def main():
    log.info(f"Starting Stub MCP Server on {HOST}:{PORT}...")
    server = await asyncio.start_server(lambda: StubMCPServerProtocol(), HOST, PORT)

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stub MCP Server shutting down.")
