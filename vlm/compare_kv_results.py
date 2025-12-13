import json
import os

def load_results(config_name):
    filename = f"kv_cache_results_{config_name}.json"
    if os.path.exists(filename):
        with open(filename) as f:
            return json.load(f)
    return None

def compare_results():
    configs = ["no_prefix", "prefix_caching", "chunked_prefill", "default"]
    
    results = {}
    for config in configs:
        data = load_results(config)
        if data:
            results[config] = data
    
    if not results:
        print("No results found. Run benchmarks first.")
        return
    
    print("\n" + "=" * 80)
    print("KV Cache Configuration Comparison")
    print("=" * 80)
    
    # 表头
    print(f"\n{'Config':<20} {'Avg TTFT (ms)':<15} {'Decode (tok/s)':<15} {'Prefix Gain':<15} {'Memory (MiB)':<12}")
    print("-" * 80)
    
    # 获取 baseline 用于计算提升
    baseline_ttft = None
    if "no_prefix" in results:
        baseline_ttft = results["no_prefix"]["baseline"]["avg_ttft"]
    
    for config, data in results.items():
        ttft = data["baseline"]["avg_ttft"] * 1000
        decode = data["baseline"]["avg_decode_speed"]
        prefix_gain = data["prefix_caching"]["ttft_reduction_pct"]
        memory = data["memory"]["memory_after"]
        
        # 相对于 no_prefix baseline 的提升
        ttft_improvement = ""
        if baseline_ttft and config != "no_prefix":
            baseline_val = baseline_ttft * 1000
            improvement = (baseline_val - ttft) / baseline_val * 100
            ttft_improvement = f" ({improvement:+.1f}%)"
        
        print(f"{config:<20} {ttft:>6.2f}{ttft_improvement:<8} {decode:>10.2f}      {prefix_gain:>6.1f}%         {memory}")
    
    # 序列长度对比
    print("\n" + "=" * 80)
    print("Sequence Length Impact")
    print("=" * 80)
    
    for config, data in results.items():
        print(f"\n📊 {config.upper()}:")
        seq_data = data.get("sequence_length", {})
        
        print(f"  {'Type':<10} {'Prompt Len':<12} {'TTFT (ms)':<12} {'Decode (tok/s)':<15} {'Output Tokens':<12}")
        print(f"  {'-'*60}")
        
        for length_type in ["short", "medium", "long"]:
            if length_type in seq_data:
                d = seq_data[length_type]
                print(f"  {length_type:<10} {d['prompt_length']:<12} {d['avg_ttft']*1000:<12.2f} {d['avg_decode_speed']:<15.2f} {d['avg_output_tokens']:<12.1f}")
    
    # Prefix Caching 效果对比
    print("\n" + "=" * 80)
    print("Prefix Caching Effectiveness")
    print("=" * 80)
    
    print(f"\n{'Config':<20} {'First TTFT (ms)':<18} {'Subsequent TTFT (ms)':<22} {'Reduction':<12}")
    print("-" * 80)
    
    for config, data in results.items():
        pc = data["prefix_caching"]
        first = pc["first_ttft"] * 1000
        subsequent = pc["subsequent_avg_ttft"] * 1000
        reduction = pc["ttft_reduction_pct"]
        print(f"{config:<20} {first:>10.2f}        {subsequent:>12.2f}            {reduction:>6.1f}%")
    
    # 保存对比结果
    comparison = {
        "configs": list(results.keys()),
        "summary": {}
    }
    
    for config, data in results.items():
        comparison["summary"][config] = {
            "avg_ttft_ms": data["baseline"]["avg_ttft"] * 1000,
            "avg_decode_speed": data["baseline"]["avg_decode_speed"],
            "prefix_cache_reduction_pct": data["prefix_caching"]["ttft_reduction_pct"],
            "memory_used_mib": data["memory"]["memory_after"]
        }
    
    with open("kv_cache_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ Comparison saved to kv_cache_comparison.json")
    print("=" * 80)

if __name__ == "__main__":
    compare_results()