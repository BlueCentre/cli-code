# chuk_mcp/chuk_mcp.mcp_client/messages/ping/send_messages.py
import logging

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from mcp_code.mcp_client.messages.message_method import MessageMethod

# mcp_code imports (refactored from chuk_mcp)
from mcp_code.mcp_client.messages.send_message import send_message


async def send_ping(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    timeout: float = 5.0,
    retries: int = 3,
) -> bool:
    """
    Send a ping message to the server and return success status.

    Args:
        read_stream: Stream to read responses from
        write_stream: Stream to write requests to
        timeout: Timeout in seconds for the ping response
        retries: Number of retry attempts

    Returns:
        bool: True if ping was successful, False otherwise
    """
    # Let send_message generate a unique ID
    try:
        # send the message
        response = await send_message(
            read_stream=read_stream,
            write_stream=write_stream,
            method=MessageMethod.PING,
            params=None,  # Ping doesn't require parameters
            timeout=timeout,
            retries=retries,
        )

        # Return True if we got a response (regardless of content)
        return response is not None
    except Exception as e:
        # Log exception as debug
        logging.debug(f"Ping failed: {e}")

        # failed
        return False
