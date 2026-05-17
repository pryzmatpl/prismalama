#!/usr/bin/env bash
# Benchmark qwen3.6:35b across streaming configs. Restarts server per config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${BINARY:-$ROOT/prismalama-ollama}"
MODEL="${MODEL:-qwen3.6:35b}"
MODELS_DIR="${OLLAMA_MODELS:-/home/models}"
LIB_PATH="${OLLAMA_LIBRARY_PATH:-$ROOT/build/lib/ollama/cuda:$ROOT/build/lib/ollama}"
OUT="${OUT:-$ROOT/scripts/benchmark-baselines-qwen36.json}"
EPOCHS="${EPOCHS:-2}"
NUM_PREDICT="${NUM_PREDICT:-32}"
NUM_CTX="${NUM_CTX:-2048}"
LOG_DIR="${LOG_DIR:-/tmp/prismalama-bench}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$BINARY" ]]; then
  echo "missing binary: $BINARY" >&2
  exit 1
fi

stop_server() {
  pkill -f "$BINARY serve" 2>/dev/null || true
  pkill -f "$BINARY runner" 2>/dev/null || true
  pkill -f "prismalama-ollama serve" 2>/dev/null || true
  pkill -f "prismalama-ollama runner" 2>/dev/null || true
  for _ in $(seq 1 15); do
    if ! curl -sf --max-time 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  sleep 1
}

start_server() {
  local tag=$1
  shift
  stop_server
  export OLLAMA_MODELS="$MODELS_DIR"
  export OLLAMA_LIBRARY_PATH="$LIB_PATH"
  export OLLAMA_NEW_ENGINE=1
  export OLLAMA_MMAP_ALLOW_LOW_RAM=1
  export OLLAMA_MEMORY_POLICY="${OLLAMA_MEMORY_POLICY:-balanced}"
  export OLLAMA_LAYER_STREAMING="${OLLAMA_LAYER_STREAMING:-1}"
  export OLLAMA_STREAMING_BUDGET="${OLLAMA_STREAMING_BUDGET:-4294967296}"
  unset OLLAMA_VULKAN
  local log="$LOG_DIR/serve-$tag.log"
  "$BINARY" serve >"$log" 2>&1 &
  for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "server failed to start; see $log" >&2
  tail -20 "$log" >&2
  exit 1
}

run_bench() {
  local tag=$1
  python3 "$ROOT/scripts/benchmark-tps.py" "$MODEL" "$EPOCHS" "$NUM_PREDICT" \
    --num-ctx "$NUM_CTX" 2>&1 | tee "$LOG_DIR/bench-$tag.txt"
}

gpu_layers_from_log() {
  local log=$1
  grep -o 'offloaded [0-9]*/[0-9]* layers to GPU' "$log" 2>/dev/null | tail -1 || echo "unknown"
}

echo "[" >"$OUT"
first=1

run_config() {
  local name=$1
  local streaming=$2
  local budget=$3
  echo "=== config: $name (streaming=$streaming budget=$budget) ===" >&2
  OLLAMA_LAYER_STREAMING=$streaming OLLAMA_STREAMING_BUDGET=$budget \
    start_server "$name"
  local layers
  layers=$(gpu_layers_from_log "$LOG_DIR/serve-$name.log")
  local bench_out
  bench_out=$(mktemp)
  if run_bench "$name" >"$bench_out" 2>&1; then
  :
  else
    echo "benchmark failed for $name" >&2
  fi
  local eval_mean wall_mean load_mean vram_gb
  eval_mean=$(grep -oP 'eval TPS \(mean/min/max\): \K[0-9.]+' "$bench_out" | head -1 || echo "null")
  wall_mean=$(grep -oP 'wall TPS \(mean/min/max\): \K[0-9.]+' "$bench_out" | head -1 || echo "null")
  load_mean=$(grep -oP 'load \(mean\): \K[0-9.]+' "$bench_out" | head -1 || echo "null")
  vram_gb=$(curl -sf http://127.0.0.1:11434/api/ps 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
m=d.get('models',[])
print(f\"{m[0].get('size_vram',0)/1e9:.2f}\" if m else 'null')
" 2>/dev/null || echo "null")
  [[ $first -eq 1 ]] || echo "," >>"$OUT"
  first=0
  python3 - <<PY >>"$OUT"
import json
print(json.dumps({
  "name": "$name",
  "layer_streaming": $streaming,
  "streaming_budget_bytes": int($budget),
  "num_ctx": int($NUM_CTX),
  "num_predict": int($NUM_PREDICT),
  "epochs": int($EPOCHS),
  "gpu_layers_log": "$layers",
  "eval_tps_mean": float("$eval_mean") if "$eval_mean" != "null" else None,
  "wall_tps_mean": float("$wall_mean") if "$wall_mean" != "null" else None,
  "load_s_mean": float("$load_mean") if "$load_mean" != "null" else None,
  "size_vram_gb": float("$vram_gb") if "$vram_gb" != "null" else None,
}, indent=2))
PY
  rm -f "$bench_out"
  stop_server
}

# Baseline matrix (4070 Ti ~12GB): budget vs throughput
run_config "stream-4g-default" 1 $((4 * 1024 * 1024 * 1024))
run_config "stream-6g" 1 $((6 * 1024 * 1024 * 1024))
run_config "stream-8g" 1 $((8 * 1024 * 1024 * 1024))
run_config "stream-2g" 1 $((2 * 1024 * 1024 * 1024))

echo "]" >>"$OUT"
echo "Wrote $OUT"
