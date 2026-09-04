Minimal npu setup llm

model_id = "OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov"
model_path = "models/npu/Phi-3.5-mini-instruct-int4-gq-ov"

import openvino_genai as ov_genai

device = "NPU"
pipe = ov_genai.LLMPipeline(model_path, device)
print(pipe.generate("What is OpenVINO?", max_length=200))
