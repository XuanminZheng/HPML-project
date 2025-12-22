import asyncio
import time
import base64
import json
from typing import List, Dict
from openai import AsyncOpenAI
from datasets import load_dataset
from io import BytesIO
from PIL import Image

VLLM_ENDPOINT = "http://localhost:8000/v1"
MODEL = "openbmb/MiniCPM-V-4"


MAX_TOKENS = 256
CONCURRENCY = 16        
NUM_REQUESTS = 50      


# ================== 初始化 ==================
client = AsyncOpenAI(base_url=VLLM_ENDPOINT, api_key="token")
ds = load_dataset("lmms-lab/MME", split="test")
samples = ds.shuffle(seed=42).select(range(NUM_REQUESTS))


def encode_image(pil_img) -> str:
    if isinstance(pil_img, dict):
        if pil_img.get("bytes") is not None:
            img = Image.open(BytesIO(pil_img["bytes"])).convert("RGB")
        elif pil_img.get("path") is not None:
            img = Image.open(pil_img["path"]).convert("RGB")
        else:
            raise ValueError(f"Invalid image dict: {pil_img}")
    else:
        img = pil_img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()
# ================== 单请求 ==================
async def single_request(sample: Dict, idx: int) -> Dict:
    prompt = sample["question"]
    img_b64 = encode_image(sample["image"])
    t0 = time.perf_counter()
    first_token_time = None

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                }
            ]
        }],
        max_tokens=MAX_TOKENS,
        stream=True,   
    )

    output_tokens = 0

    async for chunk in response:
        delta = chunk.choices[0].delta

        # 跳过非 token 的 chunk
        if delta is None or delta.content is None:
            continue

        # 第一个真正生成的 token
        if first_token_time is None:
            first_token_time = time.perf_counter()

        output_tokens += 1

    t1 = time.perf_counter()

    return {
        "id": idx,
        "prompt": prompt,
        "ttft": first_token_time - t0 if first_token_time else None,
        "latency": t1 - t0,
        "output_tokens": output_tokens,
    }

# ================== 并发调度 ==================
async def run_benchmark():
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def limited_request(prompt, idx):
        async with semaphore:
            return await single_request(prompt, idx)

    start = time.perf_counter()

    results = await asyncio.gather(*[
        limited_request(prompt, i)
        for i, prompt in enumerate(samples)
    ])

    end = time.perf_counter()

    summary = {
        "num_requests": NUM_REQUESTS,
        "concurrency": CONCURRENCY,
        "total_time": end - start,
        "throughput_rps": NUM_REQUESTS / (end - start),
        "avg_ttft": sum(r["ttft"] for r in results if r["ttft"]) / len(results),
        "avg_latency": sum(r["latency"] for r in results) / len(results),
        "avg_tokens": sum(r["output_tokens"] for r in results) / len(results),
    }

    with open(f"benchmark_results_c{CONCURRENCY}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "details": results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("===== Benchmark Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

async def main():
    global CONCURRENCY

    for c in [1, 2, 4, 6, 8, 12, 16]:
        CONCURRENCY = c
        print(f"\n=== Running benchmark: concurrency={c} ===")
        await run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())