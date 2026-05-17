#!/usr/bin/env python3
"""Benchmark TPS using Ollama API eval_count/eval_duration (canonical) and wall-clock."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
if not HOST.startswith("http"):
    HOST = f"http://{HOST}"


def get(path: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(f"{HOST}{path}", timeout=timeout) as resp:
        return json.loads(resp.read())


def post(path: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def unload_all():
    try:
        ps = get("/api/ps")
        for m in ps.get("models", []):
            post("/api/generate", {"model": m["name"], "keep_alive": 0})
    except Exception:
        pass


def benchmark_model(
    model: str,
    prompt: str,
    num_predict: int,
    epochs: int,
    warmup: int = 1,
    options: dict | None = None,
) -> list[dict]:
    unload_all()
    opts = {"num_predict": num_predict, "temperature": 0}
    if options:
        opts.update(options)
    rows = []
    total_runs = warmup + epochs
    for i in range(total_runs):
        is_warmup = i < warmup
        t0 = time.perf_counter()
        try:
            r = post(
                "/api/generate",
                {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": opts,
                },
                timeout=1800,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"generate failed: {e}\n{body}") from e
        wall = time.perf_counter() - t0
        ec = int(r.get("eval_count") or 0)
        ed_ns = int(r.get("eval_duration") or 0)
        ld_ns = int(r.get("load_duration") or 0)
        pp_ns = int(r.get("prompt_eval_count") or 0) and int(r.get("prompt_eval_duration") or 0)
        ped_ns = int(r.get("prompt_eval_duration") or 0)
        pec = int(r.get("prompt_eval_count") or 0)

        eval_tps = (ec / (ed_ns / 1e9)) if ed_ns > 0 and ec > 0 else 0.0
        wall_tps = ec / wall if wall > 0 and ec > 0 else 0.0
        prefill_tps = (pec / (ped_ns / 1e9)) if ped_ns > 0 and pec > 0 else 0.0

        row = {
            "run": i + 1,
            "warmup": is_warmup,
            "eval_count": ec,
            "eval_duration_s": ed_ns / 1e9,
            "load_duration_s": ld_ns / 1e9,
            "wall_s": wall,
            "eval_tps": eval_tps,
            "wall_tps": wall_tps,
            "prefill_tps": prefill_tps,
            "total_duration_s": int(r.get("total_duration") or 0) / 1e9,
        }
        rows.append(row)
        tag = "warmup" if is_warmup else "epoch"
        print(
            f"  [{tag} {i + 1 - warmup if not is_warmup else i + 1}/{epochs if not is_warmup else warmup}] "
            f"eval_tps={eval_tps:.2f} wall_tps={wall_tps:.2f} "
            f"tokens={ec} load={row['load_duration_s']:.2f}s wall={wall:.2f}s"
        )
    return [r for r in rows if not r["warmup"]]


def summarize(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{label}: no epoch rows")
        return
    eval_tps = [r["eval_tps"] for r in rows]
    wall_tps = [r["wall_tps"] for r in rows]
    loads = [r["load_duration_s"] for r in rows]
    print(f"\n=== {label} ===")
    print(f"  eval TPS (mean/min/max): {sum(eval_tps)/len(eval_tps):.2f} / {min(eval_tps):.2f} / {max(eval_tps):.2f}")
    print(f"  wall TPS (mean/min/max): {sum(wall_tps)/len(wall_tps):.2f} / {min(wall_tps):.2f} / {max(wall_tps):.2f}")
    print(f"  load (mean): {sum(loads)/len(loads):.2f}s")


def ps_snapshot():
    try:
        ps = get("/api/ps")
        for m in ps.get("models", []):
            vram = m.get("size_vram", m.get("size", 0))
            print(
                f"  loaded: {m['name']} size={m.get('size',0)/1e9:.2f}GB "
                f"size_vram={vram/1e9:.2f}GB ctx={m.get('context_length')}"
            )
    except Exception as e:
        print(f"  ps: {e}")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    num_predict = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    prompt = sys.argv[4] if len(sys.argv) > 4 else "Explain quantum computing in detail."
    extra_opts = {}
    if "--num-ctx" in sys.argv:
        extra_opts["num_ctx"] = int(sys.argv[sys.argv.index("--num-ctx") + 1])
    if "--num-gpu" in sys.argv:
        extra_opts["num_gpu"] = int(sys.argv[sys.argv.index("--num-gpu") + 1])

    print(f"Benchmark {model} @ {HOST}")
    print(f"  epochs={epochs} num_predict={num_predict}")
    try:
        caps = get("/api/prismalama/capabilities")
        ls = caps.get("layer_streaming", {})
        env = caps.get("environment", {})
        print(f"  version={caps.get('version')} layer_streaming={ls.get('enabled')} budget={ls.get('budget_bytes')}")
        print(
            f"  env: streaming={env.get('ollama_layer_streaming')} mmap_low_ram={env.get('ollama_mmap_allow_low_ram')} "
            f"policy={env.get('ollama_memory_policy')} vulkan={env.get('ollama_vulkan')}"
        )
    except Exception as e:
        print(f"  capabilities: {e}")

    print("  process state before:")
    ps_snapshot()

    rows = benchmark_model(model, prompt, num_predict, epochs, options=extra_opts or None)
    print("  process state after:")
    ps_snapshot()
    summarize(model, rows)


if __name__ == "__main__":
    main()
