import asyncio
import time
import base64
import json
import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from io import BytesIO

from openai import AsyncOpenAI
from datasets import load_dataset
from PIL import Image



VLLM_ENDPOINT = "http://localhost:8000/v1"
MODEL = "openbmb/MiniCPM-V-4"

MAX_TOKENS = 256
NUM_REQUESTS = 50
MAX_CONCURRENCY = 16



client = AsyncOpenAI(base_url=VLLM_ENDPOINT, api_key="token")

ds = load_dataset("lmms-lab/MME", split="test")
samples = ds.shuffle(seed=42).select(range(NUM_REQUESTS))



def encode_image(pil_img) -> str:
    if isinstance(pil_img, dict):
        if pil_img.get("bytes"):
            img = Image.open(BytesIO(pil_img["bytes"])).convert("RGB")
        else:
            img = Image.open(pil_img["path"]).convert("RGB")
    else:
        img = pil_img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@dataclass(order=True)
class Request:
    sort_index: float = field(init=False, repr=False)
    idx: int
    sample: Dict
    arrival_time: float
    est_tokens: int
    priority: float = 1.0

    def compute_score(self, now, alpha=1.0, beta=1.0, gamma=1.0):
        wait = now - self.arrival_time
        score = alpha * wait + beta / self.est_tokens + gamma * self.priority
        self.sort_index = -score



class BaseScheduler:
    def __init__(self):
        self.queue: List[Request] = []

    def add(self, r: Request):
        self.queue.append(r)

    def next_batch(self):
        raise NotImplementedError


class FCFSScheduler(BaseScheduler):
    def next_batch(self):
        batch = self.queue[:MAX_CONCURRENCY]
        self.queue = self.queue[MAX_CONCURRENCY:]
        return batch


class PriorityScheduler(BaseScheduler):
    def next_batch(self):
        now = time.perf_counter()
        for r in self.queue:
            r.compute_score(now)
        self.queue.sort()
        batch = self.queue[:MAX_CONCURRENCY]
        self.queue = self.queue[MAX_CONCURRENCY:]
        return batch


class AdaptiveBatchScheduler(BaseScheduler):
    def next_batch(self):
        q = len(self.queue)
        if q >= 32:
            b = MAX_CONCURRENCY
        elif q >= 16:
            b = MAX_CONCURRENCY // 2
        else:
            b = 1
        batch = self.queue[:b]
        self.queue = self.queue[b:]
        return batch


class CombinedScheduler(BaseScheduler):
    def next_batch(self):
        now = time.perf_counter()
        for r in self.queue:
            r.compute_score(now)
        self.queue.sort()

        q = len(self.queue)
        if q >= 32:
            b = MAX_CONCURRENCY
        elif q >= 16:
            b = MAX_CONCURRENCY // 2
        else:
            b = 1

        batch = self.queue[:b]
        self.queue = self.queue[b:]
        return batch



async def run_request(req: Request):
    sample = req.sample
    img = encode_image(sample["image"])

    t0 = time.perf_counter()
    first = None
    tokens = 0

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": sample["question"]},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            ]
        }],
        max_tokens=MAX_TOKENS,
        stream=True
    )

    async for c in response:
        d = c.choices[0].delta
        if d and d.content:
            if first is None:
                first = time.perf_counter()
            tokens += 1

    t1 = time.perf_counter()

    return {
        "latency": t1 - t0,
        "ttft": (first - t0) if first else None,
        "tokens": tokens
    }



async def run_benchmark(name, scheduler):
    results = []

    now = time.perf_counter()
    for i, s in enumerate(samples):
        scheduler.add(Request(i, s, now, MAX_TOKENS))

    start = time.perf_counter()

    while scheduler.queue:
        batch = scheduler.next_batch()
        batch_res = await asyncio.gather(*[run_request(r) for r in batch])
        results.extend(batch_res)

    end = time.perf_counter()

    lat = np.array([r["latency"] * 1000 for r in results])

    summary = {
        "scheduler": name,
        "throughput": len(results) / (end - start),
        "avg_latency_ms": lat.mean(),
        "p99_latency_ms": np.percentile(lat, 99)
    }

    print(f"\n{name}")
    for k, v in summary.items():
        if k != "scheduler":
            print(f"{k}: {v:.2f}")

    return summary


# ================== main ==================
async def main():
    summaries = []

    summaries.append(await run_benchmark(
        "Default FCFS", FCFSScheduler()
    ))
    summaries.append(await run_benchmark(
        "Priority-Based", PriorityScheduler()
    ))
    summaries.append(await run_benchmark(
        "Adaptive Batch", AdaptiveBatchScheduler()
    ))
    summaries.append(await run_benchmark(
        "Combined", CombinedScheduler()
    ))

    with open("table4_results.json", "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
