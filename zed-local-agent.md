OpenAI-Compatible Endpoints

Use an OpenAI-compatible endpoint when you have a custom base URL, model ID, and API key.

You can add a custom OpenAI-compatible provider from Agent Settings with agent: open settings. Look for Add Provider in the LLM Providers section and fill in the provider name, API URL, model ID, and context window.

You can also configure the provider in your settings file:

{
  "language_models": {
    "openai_compatible": {
      "my-provider": {
        "api_url": "https://example.com/v1",
        "available_models": [
          {
            "name": "my-model",
            "display_name": "My Model",
            "max_tokens": 128000
          }
        ]
      }
    }
  }
}

By default, OpenAI-compatible models inherit these capabilities:

    tools: true
    images: false
    parallel_tool_calls: false
    prompt_cache_key: false
    chat_completions: true
    interleaved_reasoning: false
    max_tokens_parameter: false

If a model only works with the Responses API, set capabilities.chat_completions to false. Zed will use the Responses endpoint for that model.

For reasoning models (e.g. GPT-5), set reasoning_effort to the non-none effort level your endpoint supports. This enables thinking in the agent panel and tells Zed which effort to send when thinking is enabled. The provider settings UI can configure this when adding an OpenAI-compatible provider. Zed sends OpenAI-style reasoning_effort on chat-completions requests.

If the model requires the Responses API for reasoning state, set capabilities.chat_completions to false:

{
  "language_models": {
    "openai_compatible": {
      "my-provider": {
        "api_url": "https://example.com/v1",
        "available_models": [
          {
            "name": "gpt-5",
            "max_tokens": 272000,
            "reasoning_effort": "high",
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": false,
              "interleaved_reasoning": false,
              "max_tokens_parameter": false
            }
          }
        ]
      }
    }
  }
}

Valid settings values are "none", "minimal", "low", "medium", "high", "xhigh", and "max". Use "none" in settings.json when you need to force reasoning off for an endpoint; the provider setup UI exposes the non-none values for thinking-capable models. For chat-completions endpoints that should receive prior thinking back in a dedicated reasoning_content field, also set capabilities.interleaved_reasoning to true. If the endpoint expects the output-token limit as max_tokens instead of max_completion_tokens, set capabilities.max_tokens_parameter to true.

For example, a chat-completions endpoint with the maximum OpenAI-style reasoning effort, streamed thinking, and max_tokens output limits can be configured as:

{
  "language_models": {
    "openai_compatible": {
      "my-reasoning-provider": {
        "api_url": "https://example.com/v1",
        "available_models": [
          {
            "name": "reasoning-model",
            "max_tokens": 1000000,
            "max_output_tokens": 128000,
            "reasoning_effort": "max",
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": true,
              "max_tokens_parameter": true
            }
          }
        ]
      }
    }
  }
}

Enter the API key in the provider settings UI or set the generated environment variable. Do not put API keys in settings.json.
