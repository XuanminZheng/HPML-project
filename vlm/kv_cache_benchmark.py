import time
import base64
import subprocess
import statistics
import json
import os
from openai import OpenAI

# ========== 配置 ==========
API_BASE = "http://localhost:8000/v1"
MODEL = "openbmb/MiniCPM-V-4"
IMAGE_PATH = "img/1.jpg"
PROMPTS = [
    "描述这张图片",
    "这张图片中有什么物体？",
    "详细分析这张图片的内容，包括颜色、构图和主题",
]
NUM_RUNS = 5

# ========== 初始化 ==========
client = OpenAI(base_url=API_BASE, api_key="token")

with open(IMAGE_PATH, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

def get_gpu_memory():
    """获取 GPU 显存使用情况"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    used, total = map(int, result.stdout.strip().split(", "))
    return used, total

def run_inference_streaming(prompt):
    """流式推理，返回详细指标"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
    }]
    
    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    output_text = ""
    token_count = 0
    
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
        stream=True
    )
    
    for chunk in stream:
        current_time = time.perf_counter()
        if chunk.choices[0].delta.content:
            token_count += 1
            output_text += chunk.choices[0].delta.content
            if first_token_time is None:
                first_token_time = current_time
            token_times.append(current_time)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    ttft = first_token_time - start_time if first_token_time else 0
    decode_time = total_time - ttft
    decode_tokens = token_count - 1 if token_count > 1 else token_count
    decode_speed = decode_tokens / decode_time if decode_time > 0 else 0
    
    inter_token_latencies = []
    for i in range(1, len(token_times)):
        inter_token_latencies.append(token_times[i] - token_times[i-1])
    avg_itl = statistics.mean(inter_token_latencies) if inter_token_latencies else 0
    
    return {
        "total_time": total_time,
        "ttft": ttft,
        "decode_time": decode_time,
        "decode_speed": decode_speed,
        "token_count": token_count,
        "avg_inter_token_latency": avg_itl,
        "output_text": output_text
    }

def test_prefix_caching():
    """测试 Prefix Caching 效果 - 使用相同前缀的多次请求"""
    print("\n" + "=" * 60)
    print("Testing Prefix Caching Effect")
    print("=" * 60)
    
    # 相同前缀的请求（应该能利用 prefix cache）
    base_prompt = "请仔细观察这张图片，然后"
    suffix_prompts = [
        "描述图片中的主要内容",
        "分析图片的颜色搭配",
        "说明图片的构图特点",
        "识别图片中的物体",
        "描述图片的整体氛围",
    ]
    
    results = []
    print("\n📝 Testing with shared prefix prompts...")
    
    for i, suffix in enumerate(suffix_prompts):
        prompt = base_prompt + suffix
        result = run_inference_streaming(prompt)
        results.append(result)
        print(f"  Request {i+1}: TTFT={result['ttft']*1000:.1f}ms, "
              f"Decode={result['decode_speed']:.1f} tok/s")
    
    # 分析第一次 vs 后续请求的 TTFT
    first_ttft = results[0]["ttft"]
    subsequent_ttfts = [r["ttft"] for r in results[1:]]
    avg_subsequent_ttft = statistics.mean(subsequent_ttfts)
    
    print(f"\n📊 Prefix Caching Analysis:")
    print(f"  First request TTFT:      {first_ttft*1000:.2f} ms")
    print(f"  Subsequent avg TTFT:     {avg_subsequent_ttft*1000:.2f} ms")
    print(f"  TTFT reduction:          {(1 - avg_subsequent_ttft/first_ttft)*100:.1f}%")
    
    return {
        "first_ttft": first_ttft,
        "subsequent_avg_ttft": avg_subsequent_ttft,
        "ttft_reduction_pct": (1 - avg_subsequent_ttft/first_ttft) * 100,
        "all_results": [{"ttft": r["ttft"], "decode_speed": r["decode_speed"]} for r in results]
    }

def test_different_sequence_lengths():
    """测试不同序列长度对 KV Cache 的影响"""
    print("\n" + "=" * 60)
    print("Testing Different Sequence Lengths")
    print("=" * 60)
    
    prompts_by_length = {
        "short": "这是什么？",
        "medium": "请描述这张图片的主要内容，包括你看到的物体和场景。",
        "long": "请非常详细地分析这张图片。首先描述图片的整体构图和布局，然后识别并描述图片中的所有物体、人物或元素。接着分析图片的颜色搭配、光线效果和视觉风格。最后，推测这张图片可能的拍摄场景、目的和情感表达。"
    }
    
    results = {}
    
    for length_type, prompt in prompts_by_length.items():
        print(f"\n📝 Testing {length_type} prompt (len={len(prompt)})...")
        
        run_results = []
        for i in range(NUM_RUNS):
            result = run_inference_streaming(prompt)
            run_results.append(result)
            print(f"  Run {i+1}: TTFT={result['ttft']*1000:.1f}ms, "
                  f"Decode={result['decode_speed']:.1f} tok/s, "
                  f"Output tokens={result['token_count']}")
        
        # 计算统计
        ttfts = [r["ttft"] for r in run_results]
        decode_speeds = [r["decode_speed"] for r in run_results]
        token_counts = [r["token_count"] for r in run_results]
        
        results[length_type] = {
            "prompt_length": len(prompt),
            "avg_ttft": statistics.mean(ttfts),
            "avg_decode_speed": statistics.mean(decode_speeds),
            "avg_output_tokens": statistics.mean(token_counts),
            "ttft_std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0
        }
    
    print(f"\n📊 Sequence Length Analysis:")
    for length_type, data in results.items():
        print(f"  {length_type.upper()}:")
        print(f"    Prompt length:    {data['prompt_length']} chars")
        print(f"    Avg TTFT:         {data['avg_ttft']*1000:.2f} ms")
        print(f"    Avg Decode Speed: {data['avg_decode_speed']:.2f} tok/s")
        print(f"    Avg Output Tokens: {data['avg_output_tokens']:.1f}")
    
    return results

def test_memory_utilization():
    """测试不同负载下的显存使用"""
    print("\n" + "=" * 60)
    print("Testing Memory Utilization")
    print("=" * 60)
    
    mem_before = get_gpu_memory()
    print(f"  Memory before inference: {mem_before[0]} / {mem_before[1]} MiB")
    
    # 运行多次推理
    for i in range(5):
        run_inference_streaming("描述这张图片")
        mem_during = get_gpu_memory()
        print(f"  After request {i+1}: {mem_during[0]} / {mem_during[1]} MiB")
    
    mem_after = get_gpu_memory()
    
    return {
        "memory_before": mem_before[0],
        "memory_after": mem_after[0],
        "memory_total": mem_before[1],
        "memory_delta": mem_after[0] - mem_before[0]
    }

def run_full_benchmark(config_name="default"):
    """运行完整的 KV Cache 基准测试"""
    print("\n" + "=" * 60)
    print(f"KV Cache Benchmark - Config: {config_name}")
    print("=" * 60)
    
    results = {
        "config_name": config_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # 1. 基础性能测试
    print("\n[1/4] Running baseline performance test...")
    baseline_results = []
    for i in range(NUM_RUNS):
        result = run_inference_streaming("描述这张图片")
        baseline_results.append(result)
        print(f"  Run {i+1}/{NUM_RUNS}: TTFT={result['ttft']*1000:.1f}ms, "
              f"Decode={result['decode_speed']:.1f} tok/s")
    
    results["baseline"] = {
        "avg_ttft": statistics.mean([r["ttft"] for r in baseline_results]),
        "avg_decode_speed": statistics.mean([r["decode_speed"] for r in baseline_results]),
        "avg_total_time": statistics.mean([r["total_time"] for r in baseline_results]),
    }
    
    # 2. Prefix Caching 测试
    print("\n[2/4] Running prefix caching test...")
    results["prefix_caching"] = test_prefix_caching()
    
    # 3. 序列长度测试
    print("\n[3/4] Running sequence length test...")
    results["sequence_length"] = test_different_sequence_lengths()
    
    # 4. 显存测试
    print("\n[4/4] Running memory utilization test...")
    results["memory"] = test_memory_utilization()
    
    # 保存结果
    output_file = f"kv_cache_results_{config_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Config: {config_name}")
    print(f"  Avg TTFT: {results['baseline']['avg_ttft']*1000:.2f} ms")
    print(f"  Avg Decode Speed: {results['baseline']['avg_decode_speed']:.2f} tok/s")
    print(f"  Prefix Cache TTFT Reduction: {results['prefix_caching']['ttft_reduction_pct']:.1f}%")
    print(f"  Memory Used: {results['memory']['memory_after']} MiB")
    
    return results

if __name__ == "__main__":
    import sys
    config_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    run_full_benchmark(config_name)