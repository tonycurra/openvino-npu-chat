# Archive

`npu_openai_server.py` is an earlier OpenAI `/v1` adapter. It served Phi-3.5 and injected tool schemas into the prompt as text, because that model's chat template does not emit tool calls.

The current editor bridge is [`zed_server.py`](../zed_server.py). It serves Qwen3 8B, whose template does emit tool calls, and trims each request to the compiled NPU prompt budget.

Nothing in the live app imports this file. It stays here as a reference for the prompt-injection approach.
