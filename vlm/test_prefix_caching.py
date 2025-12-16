import time
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

with open("img/1.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

def test_ttft(prompt):
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model="openbmb/MiniCPM-V-4",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }],
        max_tokens=50,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            ttft = time.perf_counter() - start
            # 消费剩余流
            for _ in stream:
                pass
            return ttft
    return 0

# 相同前缀测试
prefix = "请仔细分析这张图片，然后"
prompts = [
    prefix + "描述主要内容",
    prefix + "分析颜色搭配", 
    prefix + "说明构图特点",
]

print("Testing Prefix Caching...")
print("-" * 50)

ttfts = []
for i, prompt in enumerate(prompts):
    ttft = test_ttft(prompt)
    ttfts.append(ttft)
    print(f"Request {i+1}: TTFT = {ttft*1000:.2f} ms")

print("-" * 50)
print(f"First TTFT:      {ttfts[0]*1000:.2f} ms")
print(f"Subsequent avg:  {sum(ttfts[1:])/len(ttfts[1:])*1000:.2f} ms")

reduction = (1 - sum(ttfts[1:])/len(ttfts[1:]) / ttfts[0]) * 100
print(f"Reduction:       {reduction:.1f}%")

if reduction > 10:
    print("\n✅ Prefix Caching is ENABLED")
else:
    print("\n❌ Prefix Caching is DISABLED (or not effective)")