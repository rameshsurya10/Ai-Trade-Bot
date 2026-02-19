"""
MT5 Bridge Protocol
===================
Shared message format between bridge client (Linux) and server (Windows).
Uses JSON over TCP with length-prefixed framing.

Message Format:
    [4 bytes: message length (big-endian)] + [JSON payload]

Request:
    {"id": "uuid", "method": "copy_rates_from_pos", "args": [...], "kwargs": {...}}

Response:
    {"id": "uuid", "success": true, "data": ..., "error": null}
"""

import json
import struct
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


HEADER_SIZE = 4  # 4 bytes for message length (uint32 big-endian)
MAX_MESSAGE_SIZE = 50 * 1024 * 1024  # 50 MB max message size


@dataclass
class MT5Request:
    """Request from client to server."""
    id: str
    method: str
    args: List[Any]
    kwargs: Dict[str, Any]

    def to_bytes(self) -> bytes:
        """Serialize to length-prefixed JSON bytes."""
        payload = json.dumps(asdict(self)).encode('utf-8')
        header = struct.pack('>I', len(payload))
        return header + payload


@dataclass
class MT5Response:
    """Response from server to client."""
    id: str
    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_bytes(self) -> bytes:
        """Serialize to length-prefixed JSON bytes."""
        payload = json.dumps(asdict(self)).encode('utf-8')
        header = struct.pack('>I', len(payload))
        return header + payload


def read_message(sock) -> Optional[dict]:
    """
    Read a length-prefixed JSON message from a socket.

    Args:
        sock: Socket to read from

    Returns:
        Parsed dict or None on connection close
    """
    # Read header (4 bytes)
    header = _recv_exact(sock, HEADER_SIZE)
    if not header:
        return None

    msg_len = struct.unpack('>I', header)[0]
    if msg_len > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {msg_len} bytes")

    # Read payload
    payload = _recv_exact(sock, msg_len)
    if not payload:
        return None

    return json.loads(payload.decode('utf-8'))


def _recv_exact(sock, n: int) -> Optional[bytes]:
    """Receive exactly n bytes from socket."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)
