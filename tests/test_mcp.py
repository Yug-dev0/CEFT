"""Unit tests for Ceft's Model Context Protocol (MCP) server."""

import json
from unittest.mock import MagicMock, patch

from app.mcp import handle_request


def test_mcp_initialize():
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    }
    resp = handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "ceft"
    assert "tools" in resp["result"]["capabilities"]


def test_mcp_ping():
    req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    resp = handle_request(req)
    assert resp["id"] == 2
    assert resp["result"] == {}


def test_mcp_notification():
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = handle_request(req)
    assert resp is None


def test_mcp_tools_list():
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    resp = handle_request(req)
    assert resp["id"] == 3
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "ceft_search" in tool_names
    assert "ceft_ask" in tool_names
    assert "ceft_get_evidence" in tool_names
    assert "ceft_get_entity" in tool_names
    assert "ceft_list_sources" in tool_names
    assert "ceft_get_timeline" in tool_names


def test_mcp_unknown_tool():
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}},
    }
    resp = handle_request(req)
    assert resp["id"] == 4
    assert resp["result"]["isError"] is True
    assert "Unknown tool" in resp["result"]["content"][0]["text"]


def test_mcp_unknown_method():
    req = {"jsonrpc": "2.0", "id": 5, "method": "unknown_method"}
    resp = handle_request(req)
    assert resp["id"] == 5
    assert "error" in resp
    assert resp["error"]["code"] == -32601


@patch("app.mcp.retrieval.retrieve")
def test_mcp_tool_search(mock_retrieve):
    mock_res = MagicMock()
    mock_res.query = "checkout timeout"
    mock_res.subqueries = []
    mock_res.hits = [
        {
            "id": "seg-1",
            "source": "incident.mp4",
            "timestamp": "00:10 - 00:20",
            "score": 0.95,
            "transcript": "fixed timeout",
            "visual_summary": "slide showing backoff",
            "entities": ["Checkout"],
            "speakers": ["Alice"],
            "metadata": {"modality": "video"},
            "via": None,
        }
    ]
    mock_retrieve.return_value = mock_res

    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "ceft_search", "arguments": {"query": "checkout timeout"}},
    }
    resp = handle_request(req)
    assert resp["id"] == 6
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert data["query"] == "checkout timeout"
    assert len(data["hits"]) == 1
    assert data["hits"][0]["source"] == "incident.mp4"


@patch("app.mcp.retrieval.retrieve")
@patch("app.mcp.synthesize_answer")
def test_mcp_tool_ask(mock_synth, mock_retrieve):
    mock_retrieve.return_value.hits = []
    mock_answer = MagicMock()
    mock_answer.answer = "The team resolved it with backoff [1]."
    mock_answer.grounded = True
    mock_answer.method = "llm"
    mock_answer.sources = [{"citation": 1, "source": "incident.mp4"}]
    mock_synth.return_value = mock_answer

    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "ceft_ask", "arguments": {"query": "How was it fixed?"}},
    }
    resp = handle_request(req)
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert data["grounded"] is True
    assert "backoff [1]" in data["answer"]


def test_mcp_tool_list_sources():
    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "ceft_list_sources", "arguments": {}},
    }
    resp = handle_request(req)
    assert resp["id"] == 8
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert isinstance(data, list)


def test_mcp_tool_get_timeline():
    req = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {"name": "ceft_get_timeline", "arguments": {"limit": 10}},
    }
    resp = handle_request(req)
    assert resp["id"] == 9
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert isinstance(data, list)
