import openvino_genai as ov_genai

model_path = "models/npu/Phi-3.5-mini-instruct-int4-gq-ov"
device = "NPU"

pipe = ov_genai.LLMPipeline(model_path, device)

while True:
    try:
        prompt = input("> ")
    except EOFError:
        break
    if not prompt:
        continue
    print(pipe.generate(prompt))
