#!/usr/bin/env python3
"""
OpenAI-compatible HTTP server for a local OpenVINO model running on the NPU.

WHY THIS EXISTS
---------------
Agent clients (Hermes, Zed, Continue, anything speaking the OpenAI API) expect a
lot more than "send prompt, get string back". They expect:

  * the FULL conversation replayed on every request (they hold the memory, not us)
  * Server-Sent Event streaming, because most agent loops set stream=true
  * a /v1/models endpoint, which many clients probe on startup
  * a complete response envelope (id/object/created/usage/finish_reason)
  * tool calls returned as a structured `tool_calls` array

The 4-line `LLMPipeline` snippet gives you none of that. This file is the
adapter between that snippet and the OpenAI wire format.

THE TOOL-CALLING CAVEAT (read this)
-----------------------------------
Phi-3.5's bundled chat_template.jinja has NO tools support -- `grep -c tools
chat_template.jinja` returns 0. So OpenVINO's native `ChatHistory.set_tools()`
would be silently dropped by the template.

Instead we do prompt-injected tool calling: tool schemas are appended to the
system message as text, the model is asked to emit a bare JSON object, and we
regex that back out into a proper `tool_calls` array.

This is best-effort. Phi-3.5-mini-int4 is a small quantized model and is not a
reliable tool caller -- expect it to sometimes ignore the format, wrap JSON in
prose, or hallucinate tool names. If tool use needs to actually work, serve a
tool-tuned model (Qwen3-8B) via OpenVINO Model Server, which ships real grammar-
constrained tool parsers including hermes3.

THE CONTEXT CAVEAT
------------------
The NPU has a hard prompt-length ceiling far below Phi-3.5's nominal 128K.
MAX_PROMPT_TOKENS below is the budget; older turns are dropped to fit. Agent
system prompts are long, so you may have very little room left for actual
conversation. Raise it only if your NPU tolerates it.

REQUIREMENTS
------------
    pip install openvino-genai fastapi uvicorn

USAGE
-----
    python npu_openai_server.py

Then point the client at:
    Base URL : http://localhost:8000/v1
    API key  : any non-empty string (this server does not authenticate)
    Model    : phi-3.5-mini
"""

import json
import queue
import re
import threading
import time
import uuid

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "models/npu/Phi-3.5-mini-instruct-int4-gq-ov"
DEVICE = "NPU"  # "NPU", "GPU", or "CPU" -- CPU is a useful fallback for debugging

# Advertised model id. Clients send this back in requests; we ignore the value
# and always serve the one loaded pipeline, but it must be non-empty for
# clients that validate it.
MODEL_NAME = "phi-3.5-mini"

# Default cap on generated tokens when the client does not specify one.
MAX_NEW_TOKENS = 512

# Prompt budget. Oldest conversation turns are dropped until the rendered
# prompt fits under this. Tuned for the NPU's limited prompt length.
MAX_PROMPT_TOKENS = 1800

HOST = "0.0.0.0"
PORT = 8000

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

# Populated by load_model(). These are deliberately NOT loaded at import time:
# openvino spins up a multiprocessing forkserver which re-imports this module,
# so a module-level load would compile the model onto the NPU twice and burn
# double the memory.
pipe = None
tokenizer = None

# The NPU pipeline is single-threaded: concurrent generate() calls corrupt
# state. Every generation acquires this lock, so requests queue up and run
# one at a time.
pipe_lock = threading.Lock()


def load_model():
    """Compile the model onto the target device. Call once, from __main__."""
    global pipe, tokenizer
    print(f"Loading {MODEL_PATH} on {DEVICE}... (first load compiles and is slow)")
    pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)
    tokenizer = pipe.get_tokenizer()
    print(f"Model ready. Serving OpenAI API at http://{HOST}:{PORT}/v1")

app = FastAPI(title="OpenVINO NPU OpenAI Bridge")


# ---------------------------------------------------------------------------
# Tool calling
#
# Injected into the system prompt because the Phi chat template ignores tools.
# ---------------------------------------------------------------------------

TOOL_PREAMBLE = """You have access to the following tools:

{tools}

To use a tool, respond with ONLY a JSON object in exactly this form, and nothing else:
{{"name": "<tool name>", "arguments": {{<arguments object>}}}}

If no tool is needed, answer normally in plain text."""

# Wrappers small models like to put around the JSON. Stripped before scanning.
TOOL_WRAPPER_RE = re.compile(r"</?tool_call>|```(?:json)?", re.IGNORECASE)

# Incremental JSON parser. Used instead of a regex because a regex cannot match
# balanced braces -- `arguments` is itself an object, so any non-greedy pattern
# stops at the inner closing brace and yields invalid JSON.
JSON_DECODER = json.JSONDecoder()


def build_tool_prompt(tools):
    """Render OpenAI tool schemas into the injected system-prompt block."""
    # OpenAI nests the real schema under "function"; older clients send it flat.
    described = [json.dumps(t.get("function", t), ensure_ascii=False) for t in tools]
    return TOOL_PREAMBLE.format(tools="\n".join(described))


def parse_tool_call(text, valid_names):
    """
    Try to read a tool call out of the model's raw reply.

    `valid_names` is the set of tool names the client actually declared; any
    other name is a hallucination and is ignored, since returning it would hand
    the client a call to a function it cannot execute.

    Returns an OpenAI-shaped `tool_calls` list, or None if the reply does not
    look like a tool call (in which case it is treated as ordinary content).
    """
    cleaned = TOOL_WRAPPER_RE.sub(" ", text)

    # Try to decode a JSON object starting at each '{'. raw_decode handles
    # nesting correctly and ignores trailing prose, which matters because the
    # model often emits a valid call and then keeps talking about it.
    for start in (i for i, ch in enumerate(cleaned) if ch == "{"):
        try:
            payload, _ = JSON_DECODER.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict) or "name" not in payload:
            continue
        if valid_names and payload["name"] not in valid_names:
            continue

        arguments = payload.get("arguments", {})
        # Some replies put arguments as a pre-encoded JSON string.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        return [{
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": str(payload["name"]),
                # OpenAI requires arguments as a JSON *string*, not an object.
                "arguments": json.dumps(arguments),
            },
        }]

    return None


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def normalize(messages, tools):
    """
    Convert an OpenAI `messages` array into what the Phi chat template accepts.

    The template only understands system/user/assistant with string content, so
    this flattens everything else down to those three roles.
    """
    out = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Vision-style content arrives as a list of parts; keep the text ones.
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        content = content or ""

        # Tool results have no role in the template -- fold them into the
        # conversation as user-visible text so the model can still see them.
        if role == "tool":
            name = msg.get("name", "tool")
            out.append({"role": "user", "content": f"Result of {name}: {content}"})
            continue

        # An assistant turn that made a tool call has content=None; replay the
        # call itself so the model retains what it just did.
        if role == "assistant" and msg.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": json.dumps(msg["tool_calls"], ensure_ascii=False),
            })
            continue

        if role not in ("system", "user", "assistant"):
            role = "user"

        out.append({"role": role, "content": content})

    # Append tool instructions to the system message, or create one.
    if tools:
        preamble = build_tool_prompt(tools)
        if out and out[0]["role"] == "system":
            out[0] = {
                "role": "system",
                "content": out[0]["content"] + "\n\n" + preamble,
            }
        else:
            out.insert(0, {"role": "system", "content": preamble})

    return out


def count_tokens(messages):
    """
    Token length of these messages once rendered through the chat template.

    encode() only accepts strings, so the history is rendered to the actual
    prompt text first -- this counts the template's control tokens too, which
    is what the NPU's prompt limit actually sees.
    """
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return len(tokenizer.encode(prompt).input_ids.data[0])


def trim(messages):
    """
    Drop the oldest turns until the prompt fits MAX_PROMPT_TOKENS.

    The system message is always preserved -- for an agent it carries the
    instructions and tool definitions, so losing it breaks everything.
    """
    has_system = bool(messages) and messages[0]["role"] == "system"
    system = messages[0] if has_system else None
    rest = messages[1:] if has_system else messages

    while rest:
        candidate = ([system] + rest) if system else rest
        try:
            if count_tokens(candidate) <= MAX_PROMPT_TOKENS:
                return candidate
        except Exception as exc:
            # Tokenizer failure shouldn't block the request, but it means the
            # prompt budget is NOT being enforced -- say so loudly.
            print(f"WARNING: token counting failed, skipping trim: {exc}")
            return candidate
        rest = rest[1:]  # drop the oldest non-system message and retry

    # Everything but the system prompt was dropped and it still doesn't fit.
    # Send it anyway -- the pipeline will report the real limit.
    return [system] if system else messages[-1:]


# ---------------------------------------------------------------------------
# Response envelope helpers
# ---------------------------------------------------------------------------


def make_id():
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def unary_response(text, tool_calls, prompt_tokens, completion_tokens):
    """Build a complete non-streaming chat.completion object."""
    # OpenAI sets content to null when the turn is a tool call.
    message = {"role": "assistant", "content": None if tool_calls else text}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": make_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
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
    """Frame a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(payload)}\n\n"


def stream_chunk(chunk_id, created, delta, finish_reason=None):
    """Build one chat.completion.chunk for the SSE stream."""
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models():
    """Model discovery. Many clients call this on startup and fail without it."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()

    messages = data.get("messages", [])
    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)

    tools = data.get("tools") or []
    tool_names = {t.get("function", t).get("name") for t in tools}

    # Clients send either max_tokens (legacy) or max_completion_tokens (current).
    max_tokens = (
        data.get("max_tokens")
        or data.get("max_completion_tokens")
        or MAX_NEW_TOKENS
    )

    prepared = trim(normalize(messages, tools))
    history = ov_genai.ChatHistory(prepared)

    # max_new_tokens (not max_length) so the cap applies to generated tokens
    # only, independent of how long the prompt happens to be.
    config = {"max_new_tokens": int(max_tokens)}

    temperature = data.get("temperature")
    if temperature is not None:
        config["temperature"] = float(temperature)
        # OpenVINO needs sampling switched on explicitly; temp=0 means greedy.
        config["do_sample"] = float(temperature) > 0
    if data.get("top_p") is not None:
        config["top_p"] = float(data["top_p"])

    if data.get("stream"):
        return StreamingResponse(
            stream_generate(history, config, tool_names),
            media_type="text/event-stream",
        )

    with pipe_lock:
        text = str(pipe.generate(history, **config)).strip()

    tool_calls = parse_tool_call(text) if tools else None

    try:
        prompt_tokens = count_tokens(prepared)
        completion_tokens = len(tokenizer.encode(text).input_ids.data[0])
    except Exception as exc:
        print(f"WARNING: usage token counting failed: {exc}")
        prompt_tokens = completion_tokens = 0

    return JSONResponse(
        unary_response(text, tool_calls, prompt_tokens, completion_tokens)
    )


def stream_generate(history, config, has_tools):
    """
    Generator yielding SSE chunks.

    pipe.generate() is blocking and pushes tokens through a callback, so it runs
    on a worker thread and hands tokens to this generator via a queue.
    """
    chunk_id = make_id()
    created = int(time.time())
    tokens = queue.Queue()
    collected = []

    def on_token(text):
        """Streamer callback, invoked on the worker thread per token chunk."""
        tokens.put(text)
        collected.append(text)
        return ov_genai.StreamingStatus.RUNNING

    def run():
        try:
            with pipe_lock:
                pipe.generate(history, streamer=on_token, **config)
        except Exception as exc:
            # Wrapped in a dict so the consumer can tell it from a token.
            tokens.put({"error": str(exc)})
        finally:
            tokens.put(None)  # sentinel: generation finished

    threading.Thread(target=run, daemon=True).start()

    # OpenAI's first chunk announces the role.
    yield sse(stream_chunk(chunk_id, created, {"role": "assistant", "content": ""}))

    error = None
    while True:
        item = tokens.get()
        if item is None:
            break
        if isinstance(item, dict):
            error = item["error"]
            break
        # When tools are active the reply might be a tool call, which can only
        # be recognized once complete -- so buffer rather than stream partial
        # JSON that the client would render as visible garbage.
        if not has_tools:
            yield sse(stream_chunk(chunk_id, created, {"content": item}))

    if error:
        yield sse({"error": {"message": error}})
        yield "data: [DONE]\n\n"
        return

    text = "".join(collected).strip()
    tool_calls = parse_tool_call(text) if has_tools else None

    if tool_calls:
        # Streaming tool_calls require an `index` on each entry.
        yield sse(stream_chunk(chunk_id, created, {
            "tool_calls": [{"index": i, **call} for i, call in enumerate(tool_calls)]
        }))
        yield sse(stream_chunk(chunk_id, created, {}, "tool_calls"))
    else:
        # Buffered because tools were active but it turned out to be prose --
        # emit it now as a single chunk.
        if has_tools:
            yield sse(stream_chunk(chunk_id, created, {"content": text}))
        yield sse(stream_chunk(chunk_id, created, {}, "stop"))

    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host=HOST, port=PORT)
