from vllm import LLM, SamplingParams

llm = LLM('TinyLlama/TinyLlama-1.1B-Chat-v1.0', gpu_memory_utilization=0.7)
params = SamplingParams(max_tokens=128, temperature=0.7)

outputs = llm.generate(['What is special about Python (programming language)?'], params)

print(outputs[0].outputs[0].text.strip())