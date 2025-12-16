import argparse
import time
import base64
import subprocess
import statistics
import json
from openai import OpenAI

# ========== 配置 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="baseline",
                    help="config: baseline / int8 / int4")
parser.add_argument("--port", type=int, default=8000,
                    help="vLLM API port")
args = parser.parse_args()

API_BASE = f"http://localhost:{args.port}/v1"
MODEL = "openbmb/MiniCPM-V-4"
IMAGE_PATH = "img/1.jpg"
PROMPT = "描述这张图片"
NUM_RUNS = 10  # 测试次数

# ========== 初始化 ==========
client = OpenAI(base_url=API_BASE, api_key="token")

# 加载图片
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

def run_inference_streaming():
    """执行一次流式推理，返回详细时间指标"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
    }]
    
    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    output_text = ""
    token_count = 0
    
    # 流式请求
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
            
            # 记录首 token 时间
            if first_token_time is None:
                first_token_time = current_time
            
            token_times.append(current_time)
    
    end_time = time.perf_counter()
    
    # 计算各项指标
    total_time = end_time - start_time
    ttft = first_token_time - start_time if first_token_time else 0  # Time to First Token
    
    # Prefill 时间 ≈ TTFT (处理输入到生成第一个token)
    prefill_time = ttft
    
    # Decode 时间 = 总时间 - Prefill 时间
    decode_time = total_time - prefill_time
    
    # Decode 速度 (tokens/s) - 不包括第一个 token
    decode_tokens = token_count - 1 if token_count > 1 else token_count
    decode_speed = decode_tokens / decode_time if decode_time > 0 else 0
    
    # 计算 token 间隔时间
    inter_token_latencies = []
    for i in range(1, len(token_times)):
        inter_token_latencies.append(token_times[i] - token_times[i-1])
    
    avg_inter_token_latency = statistics.mean(inter_token_latencies) if inter_token_latencies else 0
    
    return {
        "total_time": total_time,
        "ttft": ttft,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "decode_speed": decode_speed,
        "token_count": token_count,
        "avg_inter_token_latency": avg_inter_token_latency,
        "output_text": output_text
    }

def run_inference_non_streaming():
    """非流式推理，获取 token 使用量"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }],
        max_tokens=512
    )
    return response.usage.prompt_tokens, response.usage.completion_tokens

def profile_performance():
    """性能分析主函数"""
    print("=" * 60)
    print("MiniCPM-V-4 Performance Profiling")
    print("=" * 60)
    
    # 预热
    print("\n[1/5] Warming up...")
    run_inference_streaming()
    
    # 获取 token 数量（用非流式获取准确值）
    print("\n[2/5] Getting token counts...")
    input_tokens, _ = run_inference_non_streaming()
    
    # 获取 GPU 显存
    print("\n[3/5] Checking GPU memory...")
    mem_used, mem_total = get_gpu_memory()
    
    # 运行多次流式测试
    print(f"\n[4/5] Running {NUM_RUNS} streaming inference tests...")
    
    all_results = []
    for i in range(NUM_RUNS):
        result = run_inference_streaming()
        all_results.append(result)
        print(f"  Run {i+1}/{NUM_RUNS}: "
              f"TTFT={result['ttft']*1000:.1f}ms, "
              f"Decode={result['decode_speed']:.1f} tok/s, "
              f"Total={result['total_time']:.2f}s")
    
    # 计算统计数据
    print("\n[5/5] Calculating metrics...")
    
    # 提取各项指标
    total_times = [r["total_time"] for r in all_results]
    ttfts = [r["ttft"] for r in all_results]
    prefill_times = [r["prefill_time"] for r in all_results]
    decode_times = [r["decode_time"] for r in all_results]
    decode_speeds = [r["decode_speed"] for r in all_results]
    token_counts = [r["token_count"] for r in all_results]
    inter_token_latencies = [r["avg_inter_token_latency"] for r in all_results]
    
    # 计算 prefill 速度 (input tokens / prefill time)
    prefill_speeds = [input_tokens / t if t > 0 else 0 for t in prefill_times]
    
    # 统计函数
    def calc_stats(data):
        return {
            "avg": statistics.mean(data),
            "min": min(data),
            "max": max(data),
            "std": statistics.stdev(data) if len(data) > 1 else 0
        }
    
    # 打印结果
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS REPORT")
    print("=" * 60)
    
    # 首响时间 (TTFT)
    ttft_stats = calc_stats(ttfts)
    print("\n️  Time to First Token (TTFT):")
    print(f"  Average:    {ttft_stats['avg']*1000:.2f} ms")
    print(f"  Min:        {ttft_stats['min']*1000:.2f} ms")
    print(f"  Max:        {ttft_stats['max']*1000:.2f} ms")
    print(f"  Std Dev:    {ttft_stats['std']*1000:.2f} ms")
    
    # Prefill 速度
    prefill_stats = calc_stats(prefill_speeds)
    print("\n Prefill Speed:")
    print(f"  Average:    {prefill_stats['avg']:.2f} tokens/s")
    print(f"  Min:        {prefill_stats['min']:.2f} tokens/s")
    print(f"  Max:        {prefill_stats['max']:.2f} tokens/s")
    print(f"  Input Tokens: {input_tokens}")
    
    # Decode 速度
    decode_stats = calc_stats(decode_speeds)
    print("\n Decode Speed:")
    print(f"  Average:    {decode_stats['avg']:.2f} tokens/s")
    print(f"  Min:        {decode_stats['min']:.2f} tokens/s")
    print(f"  Max:        {decode_stats['max']:.2f} tokens/s")
    print(f"  Std Dev:    {decode_stats['std']:.2f} tokens/s")
    
    # Inter-token Latency
    itl_stats = calc_stats(inter_token_latencies)
    print("\n Inter-Token Latency:")
    print(f"  Average:    {itl_stats['avg']*1000:.2f} ms/token")
    print(f"  Min:        {itl_stats['min']*1000:.2f} ms/token")
    print(f"  Max:        {itl_stats['max']*1000:.2f} ms/token")
    
    # 端到端延迟
    e2e_stats = calc_stats(total_times)
    print("\n End-to-End Latency:")
    print(f"  Average:    {e2e_stats['avg']:.3f} s")
    print(f"  Min:        {e2e_stats['min']:.3f} s")
    print(f"  Max:        {e2e_stats['max']:.3f} s")
    print(f"  Std Dev:    {e2e_stats['std']:.3f} s")
    
    # 吞吐量
    avg_output_tokens = statistics.mean(token_counts)
    throughput_requests = 1 / e2e_stats['avg']
    throughput_tokens = avg_output_tokens / e2e_stats['avg']
    print("\n Throughput:")
    print(f"  Requests/s: {throughput_requests:.2f}")
    print(f"  Tokens/s:   {throughput_tokens:.2f} (output)")
    
    # Token 统计
    print("\n Token Statistics:")
    print(f"  Input Tokens:     {input_tokens}")
    print(f"  Output Tokens:    {avg_output_tokens:.1f} (avg)")
    
    # GPU 显存
    print("\n GPU Memory:")
    print(f"  Used:        {mem_used} MiB")
    print(f"  Total:       {mem_total} MiB")
    print(f"  Utilization: {mem_used/mem_total*100:.1f}%")
    
    print("\n" + "=" * 60)
    
    # 返回完整结果
    results = {
        "config": getattr(args, "config", "default"),
        "api_base": API_BASE,
        "num_runs": NUM_RUNS,
        "input_tokens": input_tokens,
        "gpu_memory_used": mem_used,
        "gpu_memory_total": mem_total,
        "ttft_stats": ttft_stats,
        "total_time_stats": e2e_stats,
        "decode_speed_stats": decode_stats,
        "prefill_speed_stats": prefill_stats,
        "inter_token_latency_stats": itl_stats,
        "throughput": {
            "requests_per_sec": throughput_requests,
            "tokens_per_sec": throughput_tokens,
        },
        "tokens": {
            "input": input_tokens,
            "output_avg": avg_output_tokens,
        },
        "memory": {
            "used_mib": mem_used,
            "total_mib": mem_total,
            "utilization": mem_used / mem_total,
        },
    }

    # ============ 保存到文件 ============
    out_file = f"perf_{getattr(args, 'config', 'default')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved to {out_file}")

    return results


if __name__ == "__main__":
    results = profile_performance()