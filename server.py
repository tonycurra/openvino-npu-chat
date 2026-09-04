from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import openvino_genai as ov_genai

app = FastAPI()
model_path = "models/npu/Phi-3.5-mini-instruct-int4-gq-ov"
device = "NPU"
pipe = ov_genai.LLMPipeline(model_path, device)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])

    # Get last message text, handle both OpenAI and Zed formats
    prompt = ""
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            prompt = last.get("content", "") or last.get("text", "")
        else:
            prompt = str(last)

    if not prompt:
        return JSONResponse({"error": "No prompt"}, status_code=400)

    # Crude truncation to stay under NPU's 1024 token limit
    if len(prompt) > 2000:
        prompt = prompt[-2000:]

    response_text = pipe.generate(prompt, max_length=200)

    return JSONResponse({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            }
        }]
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
