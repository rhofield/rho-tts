"""
JSON-line protocol for main process <-> worker subprocess communication.

Messages are newline-delimited JSON objects sent over stdin/stdout.
Audio data never crosses this boundary — only file paths.
"""

import json
from typing import Any

# Request types (main -> worker)
INIT = "init"
GENERATE_SINGLE = "generate_single"
GENERATE = "generate"
CANCEL = "cancel"
SHUTDOWN = "shutdown"
PING = "ping"

# Response types (worker -> main)
READY = "ready"
RESULT = "result"
ERROR = "error"
CANCELLED = "cancelled"
PONG = "pong"


def encode_message(msg_type: str, **payload: Any) -> str:
    """Encode a message as a JSON line (with trailing newline)."""
    obj = {"type": msg_type, **payload}
    return json.dumps(obj, separators=(",", ":")) + "\n"


def decode_message(line: str) -> dict:
    """Decode a JSON line into a message dict."""
    return json.loads(line)
