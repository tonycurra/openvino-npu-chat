#!/usr/bin/env python3
"""OpenAI /v1 endpoint so Zed can use Qwen3 8B on the Intel NPU.

Zed agent settings already name the provider ``openvino-npu``. Start this, then
pick that model in Zed:

    .venv/bin/python zed_server.py

Base URL http://127.0.0.1:8000/v1
Model    Qwen3-8B-int4-cw-ov
API key  any non-empty string (not checked)

Only one NPU LLM can be resident. Stop talk.py before starting this.

The NPU graph is compiled for a fixed prompt budget (MAX_PROMPT_LEN). That
cache lives in .npucache/zed-Qwen3-8B-int4-cw-ov/ and is separate from the
shorter cache talk.py uses. The first start recompiles and can take several
minutes. Qwen3 is the coding model because its chat template actually emits
tool calls; the Phi / Mistral weights do not.
"""

import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent

MODEL_ID = "Qwen3-8B-int4-cw-ov"
MODEL_PATH = ROOT / "models" / "npu" / MODEL_ID
DEVICE = "NPU"

# Prompt tokens the compiled NPU graph can accept. Zed's agent prompt is large;
# 4096 is enough for a short coding turn (instructions, tools, one file) and
# still fits in a typical Core Ultra laptop. Older turns are dropped to stay under it.
MAX_PROMPT_LEN = 4096
# Generated tokens reserved in that same graph. Requests asking for more are capped.
MIN_RESPONSE_LEN = 1024

HOST = "127.0.0.1"
PORT = 8000

pipe = None
tokenizer = None
pipe_lock = threading.Lock()

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
JSON_DECODER = json.JSONDecoder()

app = FastAPI(title="OpenVINO NPU for Zed")


def load_model():
    """Compile Qwen3 onto the NPU. Call once, from __main__ only.

    OpenVINO's forkserver re-imports this module. A load at import time would
    compile the model twice.
    """
    global pipe, tokenizer
    if not (MODEL_PATH / "openvino_model.bin").exists():
        raise SystemExit(f"missing weights: {MODEL_PATH}")
    cache = ROOT / ".npucache" / f"zed-{MODEL_ID}"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID} on {DEVICE}")
    print(f"  prompt budget {MAX_PROMPT_LEN} tokens, up to {MIN_RESPONSE_LEN} new")
    print(f"  cache {cache}")
    print("  first compile for this budget is slow; later starts reuse the cache")
    pipe = ov_genai.LLMPipeline(
        str(MODEL_PATH),
        DEVICE,
        MAX_PROMPT_LEN=MAX_PROMPT_LEN,
        MIN_RESPONSE_LEN=MIN_RESPONSE_LEN,
        GENERATE_HINT="BEST_PERF",
        CACHE_DIR=str(cache),
    )
    tokenizer = pipe.get_tokenizer()
    print(f"Ready. Zed base URL: http://{HOST}:{PORT}/v1  model: {MODEL_ID}")


def flatten_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or "")
        return "".join(parts)
    return str(content)


def normalize(messages):
    """OpenAI messages → roles the Qwen3 chat template understands."""
    out = []
    for msg in messages:
        role = msg.get("role") or "user"
        content = flatten_content(msg.get("content"))
        if role == "tool":
            out.append({"role": "tool", "content": content or "(empty)"})
            continue
        if role == "assistant":
            item = {"role": "assistant", "content": content}
            if msg.get("tool_calls"):
                item["tool_calls"] = msg["tool_calls"]
            out.append(item)
            continue
        if role not in ("system", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": content})

    systems = [m["content"] for m in out if m["role"] == "system" and m["content"]]
    rest = [m for m in out if m["role"] != "system"]
    if systems:
        return [{"role": "system", "content": "\n\n".join(systems)}] + rest
    return rest


def make_history(messages, tools):
    history = ov_genai.ChatHistory(messages)
    if tools:
        history.set_tools(tools)
    # Empty <think> block in the prompt, so the reply is the answer or a tool call.
    history.set_extra_context({"enable_thinking": False})
    return history


def count_tokens(messages, tools) -> int:
    prompt = tokenizer.apply_chat_template(
        make_history(messages, tools), add_generation_prompt=True
    )
    return len(tokenizer.encode(prompt).input_ids.data[0])


def _drop_oldest(rest):
    """Drop the oldest turn. A tool call and its results leave together."""
    dropped = rest.pop(0)
    if dropped.get("role") == "assistant" and dropped.get("tool_calls"):
        while rest and rest[0].get("role") == "tool":
            rest.pop(0)
    while rest and rest[0].get("role") == "tool":
        rest.pop(0)


def trim(messages, tools):
    """Keep the system prompt and the newest turns that fit MAX_PROMPT_LEN."""
    system = messages[0] if messages and messages[0]["role"] == "system" else None
    rest = messages[1:] if system else list(messages)
    limit = MAX_PROMPT_LEN - 8

    def packed(system_msg, rest_msgs):
        candidate = ([system_msg] + rest_msgs) if system_msg else rest_msgs
        return candidate

    while rest:
        candidate = packed(system, rest)
        try:
            if count_tokens(candidate, tools) <= limit:
                return candidate
        except Exception as exc:
            print(f"WARNING: token count failed, sending untrimmed: {exc}")
            return candidate
        _drop_oldest(rest)

    if system is None:
        return messages[-1:]

    note = "\n[truncated to fit the NPU prompt limit]"
    content = system["content"]
    lo, hi = 0, len(content)
    best = {"role": "system", "content": note}
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = {"role": "system", "content": content[:mid] + note}
        try:
            ok = count_tokens([trial], tools) <= limit
        except Exception as exc:
            print(f"WARNING: token count failed while truncating system: {exc}")
            return [system]
        if ok:
            best = trial
            lo = mid + 1
        else:
            hi = mid - 1
    if best["content"] == note and count_tokens([best], tools) > limit:
        raise RuntimeError(
            f"system prompt does not fit the {MAX_PROMPT_LEN}-token NPU budget"
        )
    print("WARNING: system prompt truncated to fit the NPU prompt budget")
    return [best]


def parse_tool_calls(text, valid_names):
    """Pull Qwen <tool_call> JSON blocks out of a finished reply."""
    visible = THINK_RE.sub("", text).strip()
    calls = []
    cursor = 0
    lowered = visible.lower()
    first = None
    while True:
        start = lowered.find("<tool_call>", cursor)
        if start < 0:
            break
        if first is None:
            first = start
        brace = visible.find("{", start)
        if brace < 0:
            break
        try:
            payload, end = JSON_DECODER.raw_decode(visible, brace)
        except json.JSONDecodeError:
            cursor = start + len("<tool_call>")
            continue
        cursor = end
        if not isinstance(payload, dict) or "name" not in payload:
            continue
        name = str(payload["name"])
        if valid_names and name not in valid_names:
            continue
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        })

    if not calls:
        return visible, None
    prose = visible[:first].strip() if first else ""
    return (prose or None), calls


def result_text(result) -> str:
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0])
    return str(result)


def completion_limit(data) -> int:
    raw = data.get("max_tokens")
    if raw is None:
        raw = data.get("max_completion_tokens")
    if raw is None:
        raw = MIN_RESPONSE_LEN
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = MIN_RESPONSE_LEN
    return max(1, min(n, MIN_RESPONSE_LEN))


def generation_config(data) -> dict:
    config = {"max_new_tokens": completion_limit(data)}
    temperature = data.get("temperature")
    if temperature is None or float(temperature) <= 0:
        config["do_sample"] = False
    else:
        config["do_sample"] = True
        config["temperature"] = float(temperature)
    if data.get("top_p") is not None:
        config["top_p"] = float(data["top_p"])
    stop = data.get("stop")
    if isinstance(stop, str) and stop:
        config["stop_strings"] = {stop}
    elif isinstance(stop, list) and stop:
        config["stop_strings"] = {str(s) for s in stop if s}
    return config


def make_id():
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def usage_counts(messages, tools, text) -> tuple[int, int]:
    try:
        prompt_tokens = count_tokens(messages, tools)
        completion_tokens = len(tokenizer.encode(text or "").input_ids.data[0])
    except Exception as exc:
        print(f"WARNING: usage count failed: {exc}")
        return 0, 0
    return prompt_tokens, completion_tokens


def unary_response(text, tool_calls, prompt_tokens, completion_tokens):
    message = {"role": "assistant", "content": None if tool_calls else text}
    if tool_calls:
        if text:
            message["content"] = text
        message["tool_calls"] = tool_calls
    return {
        "id": make_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def stream_chunk(chunk_id, created, delta, finish_reason=None):
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


@app.get("/")
async def root():
    return {
        "model": MODEL_ID,
        "device": DEVICE,
        "base_url": f"http://{HOST}:{PORT}/v1",
        "max_prompt_tokens": MAX_PROMPT_LEN,
        "max_new_tokens": MIN_RESPONSE_LEN,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if pipe is None:
        return JSONResponse(
            {"error": {"message": "model is not loaded", "type": "server_error"}},
            status_code=503,
        )
    data = await request.json()
    messages = data.get("messages") or []
    if not messages:
        return JSONResponse(
            {"error": {"message": "No messages provided", "type": "invalid_request_error"}},
            status_code=400,
        )

    tools = data.get("tools") or []
    if data.get("tool_choice") == "none":
        tools = []
    tool_names = {
        (t.get("function") or t).get("name")
        for t in tools
        if isinstance(t, dict)
    }
    tool_names.discard(None)

    try:
        prepared = trim(normalize(messages), tools)
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request_error"}},
            status_code=400,
        )

    history = make_history(prepared, tools)
    config = generation_config(data)
    print(
        f"request model={data.get('model') or MODEL_ID} "
        f"messages={len(prepared)} tools={len(tools)} "
        f"max_new={config['max_new_tokens']} stream={bool(data.get('stream'))}"
    )

    if data.get("stream"):
        return StreamingResponse(
            stream_generate(history, prepared, tools, config, tool_names),
            media_type="text/event-stream",
        )

    try:
        with pipe_lock:
            text = result_text(pipe.generate(history, **config)).strip()
    except Exception as exc:
        print(f"generate failed: {exc}")
        return JSONResponse(
            {"error": {"message": str(exc), "type": "server_error"}},
            status_code=500,
        )

    prose, tool_calls = parse_tool_calls(text, tool_names) if tools else (text, None)
    prompt_tokens, completion_tokens = usage_counts(prepared, tools, text)
    return JSONResponse(
        unary_response(prose, tool_calls, prompt_tokens, completion_tokens)
    )


def stream_generate(history, prepared, tools, config, tool_names):
    chunk_id = make_id()
    created = int(time.time())
    tokens = queue.Queue()
    collected = []

    def on_token(text):
        tokens.put(text)
        collected.append(text)
        return ov_genai.StreamingStatus.RUNNING

    def run():
        try:
            with pipe_lock:
                pipe.generate(history, streamer=on_token, **config)
        except Exception as exc:
            tokens.put({"error": str(exc)})
        finally:
            tokens.put(None)

    threading.Thread(target=run, daemon=True).start()
    yield sse(stream_chunk(chunk_id, created, {"role": "assistant", "content": ""}))

    error = None
    while True:
        item = tokens.get()
        if item is None:
            break
        if isinstance(item, dict):
            error = item["error"]
            break
        # Tool calls are only valid once the JSON is complete, so hold them.
        if not tools:
            yield sse(stream_chunk(chunk_id, created, {"content": item}))

    if error:
        print(f"generate failed: {error}")
        yield sse({"error": {"message": error}})
        yield "data: [DONE]\n\n"
        return

    text = "".join(collected).strip()
    prose, tool_calls = parse_tool_calls(text, tool_names) if tools else (text, None)
    if tool_calls:
        if prose:
            yield sse(stream_chunk(chunk_id, created, {"content": prose}))
        yield sse(stream_chunk(chunk_id, created, {
            "tool_calls": [
                {"index": i, **call} for i, call in enumerate(tool_calls)
            ]
        }))
        yield sse(stream_chunk(chunk_id, created, {}, "tool_calls"))
    else:
        if tools and prose:
            yield sse(stream_chunk(chunk_id, created, {"content": prose}))
        yield sse(stream_chunk(chunk_id, created, {}, "stop"))
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host=HOST, port=PORT)
