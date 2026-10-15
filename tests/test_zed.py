from fastapi.testclient import TestClient

import zed_server


def test_flatten_content_joins_text_parts():
    assert zed_server.flatten_content(None) == ""
    assert zed_server.flatten_content("hi") == "hi"
    assert zed_server.flatten_content(["a", {"text": "b"}, {"type": "image"}]) == "ab"


def test_normalize_merges_system_messages_and_keeps_tool_calls():
    messages = zed_server.normalize([
        {"role": "system", "content": "one"},
        {"role": "user", "content": [{"text": "ask"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "content": ""},
        {"role": "critic", "content": "note"},
        {"role": "system", "content": "two"},
    ])
    assert messages[0] == {"role": "system", "content": "one\n\ntwo"}
    assert messages[1] == {"role": "user", "content": "ask"}
    assert messages[2]["tool_calls"] == [{"id": "call_1"}]
    assert messages[3] == {"role": "tool", "content": "(empty)"}
    assert messages[4] == {"role": "user", "content": "note"}


def test_drop_oldest_removes_a_tool_call_with_its_results():
    rest = [
        {"role": "assistant", "tool_calls": [{"id": "call_1"}], "content": ""},
        {"role": "tool", "content": "a"},
        {"role": "tool", "content": "b"},
        {"role": "user", "content": "next"},
    ]
    zed_server._drop_oldest(rest)
    assert rest == [{"role": "user", "content": "next"}]


def test_drop_oldest_strips_a_leading_orphan_tool_result():
    rest = [
        {"role": "tool", "content": "stale"},
        {"role": "user", "content": "next"},
    ]
    zed_server._drop_oldest(rest)
    assert rest == [{"role": "user", "content": "next"}]


def test_trim_keeps_the_system_prompt_and_the_newest_turn(monkeypatch):
    def count(messages, tools):
        return 10 if len(messages) <= 2 else zed_server.MAX_PROMPT_LEN

    monkeypatch.setattr(zed_server, "count_tokens", count)
    trimmed = zed_server.trim([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "new"},
    ], [])
    assert trimmed[0]["content"] == "rules"
    assert trimmed[-1]["content"] == "new"
    assert all(item["content"] != "old" for item in trimmed)


def test_trim_returns_the_candidate_when_counting_fails(monkeypatch):
    def explode(messages, tools):
        raise RuntimeError("boom")

    monkeypatch.setattr(zed_server, "count_tokens", explode)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
    ]
    assert zed_server.trim(messages, []) == messages


def test_trim_shortens_a_system_prompt_that_does_not_fit(monkeypatch):
    limit = zed_server.MAX_PROMPT_LEN - 8

    def count(messages, tools):
        return len(messages[0]["content"])

    monkeypatch.setattr(zed_server, "count_tokens", count)
    trimmed = zed_server.trim([{"role": "system", "content": "x" * (limit + 500)}], [])
    assert len(trimmed) == 1
    assert trimmed[0]["role"] == "system"
    assert "truncated" in trimmed[0]["content"]
    assert len(trimmed[0]["content"]) <= limit


def test_parse_tool_calls_extracts_qwen_json_and_ignores_unknown_names():
    text = (
        "<think>plan</think>\n"
        "I'll read it.\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
        '<tool_call>{"name": "nope", "arguments": {}}</tool_call>'
    )
    prose, calls = zed_server.parse_tool_calls(text, {"read_file"})
    assert prose == "I'll read it."
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["function"]["arguments"] == '{"path": "a.py"}'


def test_parse_tool_calls_accepts_string_arguments_and_skips_broken_json():
    text = '<tool_call>not-json {"name": "echo", "arguments": "{\\"text\\": \\"hi\\"}"}'
    prose, calls = zed_server.parse_tool_calls(text, set())
    assert prose is None
    assert calls[0]["function"]["arguments"] == '{"text": "hi"}'


def test_parse_tool_calls_returns_prose_when_nothing_matches():
    prose, calls = zed_server.parse_tool_calls("just an answer", {"read_file"})
    assert prose == "just an answer"
    assert calls is None


def test_completion_limit_caps_at_the_compiled_budget():
    assert zed_server.completion_limit({}) == zed_server.MIN_RESPONSE_LEN
    assert zed_server.completion_limit({"max_tokens": 32}) == 32
    assert zed_server.completion_limit({"max_completion_tokens": 99999}) == zed_server.MIN_RESPONSE_LEN
    assert zed_server.completion_limit({"max_tokens": "nope"}) == zed_server.MIN_RESPONSE_LEN
    assert zed_server.completion_limit({"max_tokens": 0}) == 1


def test_generation_config_is_greedy_at_zero_temperature():
    greedy = zed_server.generation_config({"temperature": 0, "stop": ["END", ""]})
    assert greedy["do_sample"] is False
    assert greedy["stop_strings"] == {"END"}
    sampled = zed_server.generation_config({"temperature": 0.4, "top_p": 0.9, "stop": "END"})
    assert sampled["do_sample"] is True
    assert sampled["temperature"] == 0.4
    assert sampled["top_p"] == 0.9
    assert sampled["stop_strings"] == {"END"}


def test_unary_response_marks_tool_calls():
    body = zed_server.unary_response("ok", [{"id": "call_1"}], 3, 2)
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["id"] == "call_1"
    assert body["usage"]["total_tokens"] == 5
    assert body["model"] == zed_server.MODEL_ID


client = TestClient(zed_server.app)


def test_root_and_model_list_describe_the_loaded_id():
    root = client.get("/").json()
    assert root["model"] == zed_server.MODEL_ID
    assert root["base_url"].endswith("/v1")
    assert root["max_prompt_tokens"] == zed_server.MAX_PROMPT_LEN
    listed = client.get("/v1/models").json()
    assert listed["data"][0]["id"] == zed_server.MODEL_ID


def test_chat_requires_a_loaded_model_and_a_message(monkeypatch):
    missing = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert missing.status_code == 503
    assert zed_server.pipe is None

    monkeypatch.setattr(zed_server, "pipe", object())
    empty = client.post("/v1/chat/completions", json={"messages": []})
    assert empty.status_code == 400
