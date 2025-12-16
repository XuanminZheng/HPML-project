import asyncio
import time
import base64
import json
from openai import AsyncOpenAI
from typing import List

# ========== 配置 ==========
GPU_ENDPOINTS = [
    "http://localhost:8000/v1",  # GPU 0
    "http://localhost:8001/v1",  # GPU 1
]
MODEL = "openbmb/MiniCPM-V-4"
MAX_CONCURRENCY_PER_GPU = 8
IMAGE_PATH = "img/1.jpg"
NUM_TASKS = 50  # 任务数量

# ========== 初始化 ==========
clients = [AsyncOpenAI(base_url=endpoint, api_key="token") for endpoint in GPU_ENDPOINTS]

with open(IMAGE_PATH, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

async def single_request(client, prompt, idx, gpu_id):
    """单个请求"""
    start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }],
            max_tokens=256
        )
        latency = time.perf_counter() - start
        return {
            "idx": idx,
            "prompt": prompt,
            "output": response.choices[0].message.content,
            "gpu_id": gpu_id,
            "latency": latency,
            "success": True
        }
    except Exception as e:
        return {
            "idx": idx,
            "prompt": prompt,
            "output": "",
            "gpu_id": gpu_id,
            "latency": time.perf_counter() - start,
            "success": False,
            "error": str(e)
        }

async def gpu_worker(gpu_id, tasks, semaphore):
    """单个 GPU 的工作协程"""
    client = clients[gpu_id]
    results = []
    
    for idx, prompt in tasks:
        async with semaphore:
            result = await single_request(client, prompt, idx, gpu_id)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"  [{status}] Task {idx} on GPU {gpu_id}: {result['latency']:.2f}s")
    
    return results

async def run_inference(prompts: List[str], num_gpus: int, test_name: str):
    """运行推理测试"""
    total_tasks = len(prompts)
    
    print(f"\n{'='*60}")
    print(f"🚀 {test_name}")
    print(f"{'='*60}")
    print(f"  Tasks: {total_tasks}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Concurrency per GPU: {MAX_CONCURRENCY_PER_GPU}")
    print(f"  Total concurrency: {num_gpus * MAX_CONCURRENCY_PER_GPU}")
    print(f"{'='*60}\n")
    
    # 分配任务
    gpu_tasks = [[] for _ in range(num_gpus)]
    for idx, prompt in enumerate(prompts):
        gpu_id = idx % num_gpus
        gpu_tasks[gpu_id].append((idx, prompt))
    
    for gpu_id in range(num_gpus):
        print(f"  GPU {gpu_id}: {len(gpu_tasks[gpu_id])} tasks")
    print()
    
    # 并行执行
    semaphores = [asyncio.Semaphore(MAX_CONCURRENCY_PER_GPU) for _ in range(num_gpus)]
    start_time = time.perf_counter()
    
    all_results = await asyncio.gather(*[
        gpu_worker(gpu_id, tasks, semaphores[gpu_id])
        for gpu_id, tasks in enumerate(gpu_tasks) if gpu_id < num_gpus
    ])
    
    total_time = time.perf_counter() - start_time
    
    # 合并结果
    results = []
    for gpu_results in all_results:
        results.extend(gpu_results)
    results.sort(key=lambda x: x["idx"])
    
    # 统计
    successful = sum(1 for r in results if r["success"])
    avg_latency = sum(r["latency"] for r in results) / len(results)
    throughput = total_tasks / total_time
    
    stats = {
        "test_name": test_name,
        "num_gpus": num_gpus,
        "total_tasks": total_tasks,
        "total_time": total_time,
        "successful": successful,
        "avg_latency": avg_latency,
        "throughput": throughput,
    }
    
    print(f"\n  ⏱️  Total time: {total_time:.2f}s")
    print(f"  ✅ Successful: {successful}/{total_tasks}")
    print(f"  📊 Avg latency: {avg_latency:.2f}s")
    print(f"  🚀 Throughput: {throughput:.2f} tasks/s")
    
    return results, stats

async def main():
    # ========== 准备任务 ==========
    base_prompts = [
        "描述这张图片的主要内容",
        "图片中有什么物体？",
        "分析这张图片的颜色",
        "这张图片的构图如何？",
        "图片传达了什么情感？",
    ]
    
    prompts = []
    for i in range(NUM_TASKS // len(base_prompts) + 1):
        for prompt in base_prompts:
            if len(prompts) < NUM_TASKS:
                prompts.append(f"{prompt} (任务 {len(prompts)+1})")
    
    print(f"\n📋 Prepared {len(prompts)} tasks for benchmarking\n")
    
    # ========== 预热 ==========
    print("🔥 Warming up...")
    warmup_prompts = prompts[:2]
    await run_inference(warmup_prompts, num_gpus=2, test_name="Warmup")
    
    # ========== 单卡测试 ==========
    print("\n" + "="*60)
    print("📊 BENCHMARK: Single GPU vs Dual GPU")
    print("="*60)
    
    results_single, stats_single = await run_inference(
        prompts, 
        num_gpus=1, 
        test_name="Single GPU (GPU 0 only)"
    )
    
    # ========== 双卡测试 ==========
    results_dual, stats_dual = await run_inference(
        prompts, 
        num_gpus=2, 
        test_name="Dual GPU (GPU 0 + GPU 1)"
    )
    
    # ========== 对比结果 ==========
    print("\n" + "="*60)
    print("📈 COMPARISON RESULTS")
    print("="*60)
    
    speedup = stats_single["total_time"] / stats_dual["total_time"]
    throughput_gain = stats_dual["throughput"] / stats_single["throughput"]
    
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    Performance Comparison                    │
├─────────────────────┬───────────────┬───────────────────────┤
│       Metric        │   Single GPU  │      Dual GPU         │
├─────────────────────┼───────────────┼───────────────────────┤
│ Total Time          │ {stats_single['total_time']:>10.2f}s  │ {stats_dual['total_time']:>10.2f}s              │
│ Throughput          │ {stats_single['throughput']:>10.2f}/s │ {stats_dual['throughput']:>10.2f}/s             │
│ Avg Latency         │ {stats_single['avg_latency']:>10.2f}s  │ {stats_dual['avg_latency']:>10.2f}s              │
├─────────────────────┴───────────────┴───────────────────────┤
│                       Improvement                            │
├─────────────────────────────────────────────────────────────┤
│ Speedup:            {speedup:>6.2f}x                                   │
│ Throughput Gain:    {throughput_gain:>6.2f}x                                   │
│ Time Saved:         {stats_single['total_time'] - stats_dual['total_time']:>6.2f}s                                   │
└─────────────────────────────────────────────────────────────┘
""")
    
    # ========== 保存结果 ==========
    comparison = {
        "config": {
            "num_tasks": NUM_TASKS,
            "concurrency_per_gpu": MAX_CONCURRENCY_PER_GPU,
            "model": MODEL,
        },
        "single_gpu": stats_single,
        "dual_gpu": stats_dual,
        "comparison": {
            "speedup": speedup,
            "throughput_gain": throughput_gain,
            "time_saved": stats_single["total_time"] - stats_dual["total_time"],
        }
    }
    
    with open("gpu_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print("✅ Results saved to gpu_comparison_results.json")
    
    # ========== 结论 ==========
    print(f"""
{'='*60}
📝 CONCLUSION
{'='*60}

Using {NUM_TASKS} inference tasks:

• Single GPU completed in {stats_single['total_time']:.2f}s
• Dual GPU completed in {stats_dual['total_time']:.2f}s
• Dual GPU is {speedup:.2f}x faster than Single GPU
• Throughput improved by {(throughput_gain-1)*100:.1f}%

{'='*60}
""")

if __name__ == "__main__":
    asyncio.run(main())