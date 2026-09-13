"""Typed JSONL boundary between the Python runtime and TypeScript TUI."""

from .output import JsonlOutputSink
from .protocol import (
    ApprovalDecision,
    ClientMessage,
    JsonlProtocolError,
    ServerMessage,
    encode_message,
    parse_client_message,
    parse_server_message,
)

__all__ = [
    "ApprovalDecision",
    "ClientMessage",
    "JsonlOutputSink",
    "JsonlProtocolError",
    "ServerMessage",
    "encode_message",
    "parse_client_message",
    "parse_server_message",
]
