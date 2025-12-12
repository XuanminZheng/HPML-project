from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model='openbmb/MiniCPM-V-4',
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

params = SamplingParams(max_tokens=512, temperature=0.7)

image = Image.open("img/1.jpg")

# 使用 chat 格式
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "描述这张图片"}
        ]
    }
]

outputs = llm.chat(messages, sampling_params=params)

print(outputs[0].outputs[0].text.strip())