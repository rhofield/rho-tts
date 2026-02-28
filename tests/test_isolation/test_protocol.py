"""Tests for the JSON-line IPC protocol."""

import json

from rho_tts.isolation.protocol import (
    CANCEL,
    CANCELLED,
    ERROR,
    GENERATE,
    INIT,
    PING,
    PONG,
    READY,
    RESULT,
    SHUTDOWN,
    decode_message,
    encode_message,
)


class TestProtocolConstants:
    def test_request_types_are_strings(self):
        for name in (INIT, GENERATE, CANCEL, SHUTDOWN, PING):
            assert isinstance(name, str)

    def test_response_types_are_strings(self):
        for name in (READY, RESULT, ERROR, CANCELLED, PONG):
            assert isinstance(name, str)


class TestEncodeMessage:
    def test_basic_message(self):
        line = encode_message(PING)
        assert line.endswith("\n")
        parsed = json.loads(line)
        assert parsed["type"] == "ping"

    def test_message_with_payload(self):
        line = encode_message(INIT, provider="qwen", kwargs={"device": "cuda"})
        parsed = json.loads(line)
        assert parsed["type"] == "init"
        assert parsed["provider"] == "qwen"
        assert parsed["kwargs"] == {"device": "cuda"}

    def test_compact_json(self):
        line = encode_message(RESULT, output_path="/tmp/test.wav", success=True)
        # No spaces after separators (compact format)
        assert " " not in line.strip() or "output_path" in line

    def test_roundtrip(self):
        original = encode_message(
            GENERATE,
            texts=["Hello world"],
            output_base_path="/tmp/out",
        )
        decoded = decode_message(original)
        assert decoded["type"] == "generate"
        assert decoded["texts"] == ["Hello world"]
        assert decoded["output_base_path"] == "/tmp/out"


class TestDecodeMessage:
    def test_basic_decode(self):
        line = '{"type":"pong"}\n'
        msg = decode_message(line)
        assert msg["type"] == "pong"

    def test_decode_with_payload(self):
        line = '{"type":"ready","sample_rate":24000}\n'
        msg = decode_message(line)
        assert msg["type"] == "ready"
        assert msg["sample_rate"] == 24000

    def test_decode_error_message(self):
        line = '{"type":"error","message":"Something went wrong"}\n'
        msg = decode_message(line)
        assert msg["type"] == "error"
        assert "Something went wrong" in msg["message"]
