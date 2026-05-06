#!/usr/bin/env python3
import urllib.request, json, time, sys

def benchmark(model, tokens=256, runs=5):
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": False
    }).encode()
    
    results = []
    for i in range(runs):
        start = time.time()
        try:
            req = urllib.request.Request(
                'http://localhost:11434/v1/chat/completions',
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            elapsed = time.time() - start
            tps = result['usage']['completion_tokens'] / elapsed
            results.append(tps)
            print(f"  Run {i+1}: {tps:.1f} TPS ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  Run {i+1}: Error - {e}")
            return None
    
    avg = sum(results) / len(results)
    return avg

print("📊 GPU Inference Benchmark")
print("═" * 40)

models = ["qwen2.5:14b", "qwen3.6:27b"]
for model in models:
    print(f"\n{model}:")
    avg = benchmark(model)
    if avg:
        status = "✅ TARGET MET" if avg >= 221 else "⚠️  Below target"
        print(f"  Average: {avg:.1f} TPS {status}")

print("\n" + "═" * 40)
print("Target: 221 TPS for production")
