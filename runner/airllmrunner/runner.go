package airllmrunner

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// GPUType distinguishes AMD ROCm from NVIDIA CUDA hardware.
type GPUType string

const (
	GPUTypeAMD   GPUType = "amd"
	GPUTypeNVIDIA GPUType = "nvidia"
	GPUTypeUnknown GPUType = "unknown"
)

// GPUDevice represents a single GPU with its hardware topology metadata.
type GPUDevice struct {
	Type     GPUType // "amd", "nvidia", or "unknown"
	Index    int     // backend-native device index (HIP index on AMD, CUDA index on NVIDIA)
	BusID    string  // PCIe bus ID, e.g. "0001:3A:00.0"
	NUMANode int     // NUMA node index; -1 when unknown
}

// GPUTopology summarises all GPUs detected on the host.
type GPUTopology struct {
	Devices []GPUDevice // ordered by discovery order (AMD GPUs first, then NVIDIA)
}

// NUMANodeInfo describes a single NUMA node and the PCIe bus IDs attached to it.
type NUMANodeInfo struct {
	ID       int      // NUMA node index (0, 1, …)
	Distance int      // relative NUMA distance (0 = self)
	BusIDs   []string // PCIe bus IDs belonging to this node
}

// GPUToNVMEProximityMap maps GPU index → NVME mount paths sorted by NUMA closeness.
// The first entry in the slice is the NUMA-closest mount; ties are broken arbitrarily.
type GPUToNVMEProximityMap map[int][]string

// discoverGPUTopology probes the host for AMD ROCm and NVIDIA CUDA GPUs.
// It queries rocm-smi --json for AMD devices and nvidia-smi for NVIDIA devices,
// building a combined ordered list that the Python runner uses to set device
// visibility and affinity.
func discoverGPUTopology() GPUTopology {
	var topo GPUTopology

	// ── AMD ROCm ──────────────────────────────────────────────────────────────
	// rocm-smi --json emits: { "card$i": { "GPU ID": "...", "Bus ID": "...", ... } }
	out, err := exec.Command("rocm-smi", "--json").Output()
	if err == nil {
		var rocmMap map[string]map[string]string
		if err := json.Unmarshal(out, &rocmMap); err == nil {
			// Collect in index order (card0, card1, …)
			for i := 0; ; i++ {
				key := fmt.Sprintf("card%d", i)
				card, ok := rocmMap[key]
				if !ok {
					break
				}
				// Bus ID may be empty if the GPU is not initialised; skip it.
				busID := strings.TrimSpace(card["Bus ID"])
				if busID == "" {
					busID = "unknown"
				}
				numa := -1
				if node, ok := card["Numa Node"]; ok {
					if n, err := strconv.Atoi(strings.TrimSpace(node)); err == nil {
						numa = n
					}
				}
				dev := GPUDevice{Type: GPUTypeAMD, Index: i, BusID: busID, NUMANode: numa}
				topo.Devices = append(topo.Devices, dev)
				slog.Debug("GPU topology: discovered AMD GPU", "index", i, "bus_id", busID, "numa", numa)
			}
		}
	}

	// ── NVIDIA CUDA ────────────────────────────────────────────────────────────
	// nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader
	// Output: "0,0001:3A:00.0\n1,0001:3B:00.0\n"
	out, err = exec.Command("nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader").Output()
	if err == nil {
		reader := csv.NewReader(bytes.NewReader(out))
		records, err := reader.ReadAll()
		if err == nil {
			for _, rec := range records {
				if len(rec) < 2 {
					continue
				}
				idx, err := strconv.Atoi(strings.TrimSpace(rec[0]))
				if err != nil {
					continue
				}
				busID := strings.TrimSpace(rec[1])
				// NUMA for NVIDIA GPUs requires parsing /sys/class/nvme/nvme*.../device/numa_node
				// which is only reliable when the nvme driver is loaded. We default to -1 and
				// do not block startup on a missing sysfs node.
				numa := -1
				dev := GPUDevice{Type: GPUTypeNVIDIA, Index: idx, BusID: busID, NUMANode: numa}
				topo.Devices = append(topo.Devices, dev)
				slog.Debug("GPU topology: discovered NVIDIA GPU", "index", idx, "bus_id", busID)
			}
		}
	}

	// ── Summary log ────────────────────────────────────────────────────────────
	summary := make([]string, 0, len(topo.Devices))
	for _, d := range topo.Devices {
		summary = append(summary, fmt.Sprintf("%s:%d(%s)", d.Type, d.Index, d.BusID))
	}
	slog.Info("GPU topology discovered", "devices", summary)

	return topo
}

// discoverNUMANodes queries the host for NUMA topology using numactl --hardware
// and falls back to parsing /sys/devices/system/node/ on NUMA-aware Linux kernels.
// Returns a map of NUMA node ID → NUMANodeInfo, and an empty map if NUMA is unavailable
// (e.g., single-socket workstations, Windows, or containers without topology access).
func discoverNUMANodes() map[int]NUMANodeInfo {
	nodes := make(map[int]NUMANodeInfo)

	// ── Method 1: numactl --hardware (preferred) ─────────────────────────────────
	// Output format:
	//   available: 2 nodes (0-1)
	//   node 0 cpus: 0 1 2 3 4 5 6 7
	//   node 0 size: 257922 MB
	//   node 0 free: 223194 MB
	//   node 1 cpus: 8 9 10 11 12 13 14 15
	//   node 1 size: 258046 MB
	//   node 1 free: 234589 MB
	//   node distances:
	//   node   0   1
	//     0  10  20
	//     1  20  10
	out, err := exec.Command("numactl", "--hardware").Output()
	if err == nil {
		lines := strings.Split(string(out), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "node ") || strings.Contains(line, "distances") || strings.Contains(line, "cpus") || strings.Contains(line, "size") || strings.Contains(line, "free") {
				continue
			}
			// Parse "node 0 cpus:" or "node 0 size:" lines
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			if fields[0] != "node" {
				continue
			}
			id, err := strconv.Atoi(fields[1])
			if err != nil {
				continue
			}
			if _, ok := nodes[id]; !ok {
				nodes[id] = NUMANodeInfo{ID: id, Distance: 0}
			}
		}
		slog.Debug("NUMA nodes discovered via numactl", "count", len(nodes))
	}

	// ── Method 2: /sys/devices/system/node/ (sysfs fallback) ──────────────────
	if len(nodes) == 0 {
		sysNodeDir := "/sys/devices/system/node"
		entries, err := os.ReadDir(sysNodeDir)
		if err == nil {
			for _, entry := range entries {
				if !strings.HasPrefix(entry.Name(), "node") {
					continue
				}
				idStr := strings.TrimPrefix(entry.Name(), "node")
				id, err := strconv.Atoi(idStr)
				if err != nil {
					continue
				}
				nodes[id] = NUMANodeInfo{ID: id, Distance: 0}
				slog.Debug("NUMA node discovered via sysfs", "node", id)
			}
		}
	}

	// ── Populate BusIDs for each node via /sys/devices/system/node/nodeN/devices/ ─
	for nodeID := range nodes {
		devicesDir := fmt.Sprintf("/sys/devices/system/node/node%d/devices", nodeID)
		entries, err := os.ReadDir(devicesDir)
		if err != nil {
			continue
		}
		var busIDs []string
		for _, entry := range entries {
			// PCIe devices are listed under:
			// - /sys/devices/system/node/nodeN/devices/pci0000:00/... (PCI bridges)
			// - /sys/devices/system/node/nodeN/devices/gpu... (GPUs)
			path := filepath.Join(devicesDir, entry.Name())
			symlink, err := os.Readlink(path)
			if err != nil {
				continue
			}
			// Extract bus ID from symlink target path (e.g., ../../../pci0000:00/.../0001:3A:00.0)
			if strings.Contains(symlink, "pci") {
				parts := strings.Split(symlink, "/")
				for _, p := range parts {
					if strings.Contains(p, ":") && strings.Count(p, ":") == 2 {
						busIDs = append(busIDs, p)
					}
				}
			}
		}
		n := nodes[nodeID]
		n.BusIDs = busIDs
		nodes[nodeID] = n
	}

	// ── Summary log ─────────────────────────────────────────────────────────────
	if len(nodes) > 0 {
		var nodeIDs []int
		for id := range nodes {
			nodeIDs = append(nodeIDs, id)
		}
		slog.Info("NUMA topology discovered", "nodes", nodeIDs, "total", len(nodes))
	} else {
		slog.Info("NUMA topology not available (single-node or unsupported platform)")
	}

	return nodes
}

// mapNVMEToNUMA probes the NVME block devices under /sys/block/nvme* and resolves
// each one to its NUMA node via the PCIe topology under /sys/block/nvme*/device.
//
// Returns a map of mountPath → NUMA node ID (or -1 if unknown).
func mapNVMEToNUMA(nvmeMounts []string) map[string]int {
	result := make(map[string]int)

	// Build a quick index: for each nvmeXn1 device, look up its PCIe bus ID
	// and then find the NUMA node via the PCI device's numa_node attribute.
	// The path varies between kernels:
	//   /sys/block/nvme0n1/device/device/physfn/numa_node   (for VFIO/Virtual Function)
	//   /sys/block/nvme0n1/device/numa_node                (direct NVMe on some kernels)
	//   /sys/block/nvme0n1/device/device/numa_node          (PCIe endpoint)
	blocks, _ := filepath.Glob("/sys/block/nvme*")
	for _, blockPath := range blocks {
		blockName := filepath.Base(blockPath) // e.g. "nvme0n1"

		// Try direct numa_node first (most common on modern kernels).
		numaPath := filepath.Join(blockPath, "device", "numa_node")
		numa := readIntFile(numaPath, -1)

		// Fallback: walk up the PCIe tree to find numa_node on the physical function.
		if numa == -1 {
			// /sys/block/nvme0n1/device/device points to the parent PCIe device.
			deviceDevice := filepath.Join(blockPath, "device", "device")
			if st, err := os.Stat(deviceDevice); err == nil && st.IsDir() {
				// Try the physfn path (SR-IOV VF hierarchy).
				physfnNumaPath := filepath.Join(deviceDevice, "physfn", "numa_node")
				numa = readIntFile(physfnNumaPath, -1)
			}
			// Try the parent device directly.
			if numa == -1 {
				parentNumaPath := filepath.Join(blockPath, "device", "device", "numa_node")
				numa = readIntFile(parentNumaPath, -1)
			}
		}

		slog.Debug("NVMe block device NUMA mapping",
			"device", blockName, "numa_node", numa)

		// If we found a NUMA node for this NVMe, record it for every mount that
		// uses this device (though we don't directly track which mount maps to which
		// block device here — we just record the block → numa mapping; the caller
		// matches mounts via the provided nvmeMounts list).
		for _, mount := range nvmeMounts {
			// Simple heuristic: if the mount path looks like it might be on this NVMe,
			// associate it. In practice, nvmeMounts are filtered by the caller to
			// only include paths on NVMe filesystems.
			result[mount] = numa
		}
	}

	// Fallback: if nvmeMounts were provided but we couldn't resolve them, mark as unknown.
	for _, mount := range nvmeMounts {
		if _, ok := result[mount]; !ok {
			result[mount] = -1
		}
	}

	return result
}

// buildGPUToNVMEProximityMap sorts the provided NVME mounts for each GPU by NUMA
// proximity (closest first).  A GPU's own NUMA node has distance 0; each NUMA hop
// away adds 20 (standard ACPI SLIT distance on x86).
// The returned map's values are ordered: index 0 = NUMA-closest, last = NUMA-furthest.
// If GPU-to-NUMA mapping is unknown for a GPU, all NVMe mounts are returned in
// the order they were discovered (untouched).
func buildGPUToNVMEProximityMap(topo GPUTopology, nvmeMountToNUMA map[string]int) GPUToNVMEProximityMap {
	result := make(GPUToNVMEProximityMap)

	// Gather unique NUMA node IDs.
	numaNodes := discoverNUMANodes()

	for _, gpu := range topo.Devices {
		gpuNUMA := gpu.NUMANode
		var scored []struct {
			mount   string
			score   int
		}
		for mount, devNUMA := range nvmeMountToNUMA {
			dist := numaDistance(numaNodes, gpuNUMA, devNUMA)
			scored = append(scored, struct {
				mount string
				score int
			}{mount: mount, score: dist})
		}
		// Sort by score ascending (closest first); preserve original order on ties.
		sortByNUMADistance(scored)
		var ordered []string
		for _, s := range scored {
			ordered = append(ordered, s.mount)
		}
		result[gpu.Index] = ordered
	}

	return result
}

// numaDistance returns the NUMA distance from node a to node b using ACPI SLIT values.
// Distance to self is 10 (x86 default).  Inter-node distances are read from
// /sys/devices/system/node/nodeN/distance or fall back to 20 (one hop).
func numaDistance(nodes map[int]NUMANodeInfo, nodeA, nodeB int) int {
	if nodeA < 0 || nodeB < 0 {
		return 100 // treat unknown nodes as far
	}
	if nodeA == nodeB {
		return 10 // local NUMA node
	}
	// Read inter-node distance from sysfs (ACPI SLIT table on Linux).
	distPath := fmt.Sprintf("/sys/devices/system/node/node%d/distance", nodeA)
	data, err := os.ReadFile(distPath)
	if err != nil {
		return 20 // heuristic: one NUMA hop
	}
	// File contains space-separated distances: "10 20 20 40\n"
	fields := strings.Fields(strings.TrimSpace(string(data)))
	if nodeB < len(fields) {
		if d, err := strconv.Atoi(fields[nodeB]); err == nil {
			return d
		}
	}
	return 20 // fallback
}

// sortByNUMADistance sorts scored entries in-place by ascending NUMA distance.
func sortByNUMADistance(scored []struct {
	mount string
	score int
}) {
	// Simple insertion sort — dataset is typically ≤ 8 NVMe drives.
	for i := 1; i < len(scored); i++ {
		key := scored[i]
		j := i - 1
		for j >= 0 && scored[j].score > key.score {
			scored[j+1] = scored[j]
			j--
		}
		scored[j+1] = key
	}
}

// readIntFile reads a one-line integer from a text file, returning def on error.
func readIntFile(path string, def int) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return def
	}
	v, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return def
	}
	return v
}

// gpuVRAMBytes returns a map of GPU index → VRAM in bytes by querying rocm-smi / nvidia-smi.
// It always returns a map even when no GPUs are found.
func gpuVRAMBytes() map[int]int64 {
	vram := make(map[int]int64)

	// ── AMD ROCm ──────────────────────────────────────────────────────────────
	// rocm-smi --showmeminfo vram --json returns: { "card$i": { "vram": "16384MiB", ... } }
	out, err := exec.Command("rocm-smi", "--showmeminfo", "vram", "--json").Output()
	if err == nil {
		var m map[string]map[string]string
		if err := json.Unmarshal(out, &m); err == nil {
			for i := 0; ; i++ {
				key := fmt.Sprintf("card%d", i)
				card, ok := m[key]
				if !ok {
					break
				}
				raw, ok := card["vram"]
				if !ok {
					continue
				}
				// Format: "16384MiB" or "16384 MiB"
				raw = strings.TrimSpace(raw)
				raw = strings.ReplaceAll(raw, " ", "")
				raw = strings.TrimSuffix(raw, "MiB")
				if mb, err := strconv.ParseInt(raw, 10, 64); err == nil {
					vram[i] = mb * 1024 * 1024
				}
			}
		}
	}

	// ── NVIDIA CUDA ────────────────────────────────────────────────────────────
	// nvidia-smi --query-gpu=index,memory.total --format=csv,noheader
	// Output: "0, 16384 MiB\n1, 24576 MiB\n"
	out, err = exec.Command("nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader").Output()
	if err == nil {
		reader := csv.NewReader(bytes.NewReader(out))
		records, err := reader.ReadAll()
		if err == nil {
			for _, rec := range records {
				if len(rec) < 2 {
					continue
				}
				idx, err := strconv.Atoi(strings.TrimSpace(rec[0]))
				if err != nil {
					continue
				}
				raw := strings.TrimSpace(rec[1])
				raw = strings.ReplaceAll(raw, " ", "")
				raw = strings.TrimSuffix(raw, "MiB")
				if mb, err := strconv.ParseInt(raw, 10, 64); err == nil {
					vram[idx] = mb * 1024 * 1024
				}
			}
		}
	}

	return vram
}

// parseVRAMBudget parses a comma-separated string of GB values (e.g. "24,48,24,48")
// into a slice of byte values. Returns nil if the string is empty.
func parseVRAMBudget(raw string) []int64 {
	if raw == "" {
		return nil
	}
	var budgets []int64
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		gb, err := strconv.ParseFloat(part, 64)
		if err != nil {
			slog.Warn("AIRLLM_VRAM_BUDGET_GB: non-numeric value, skipping", "value", part)
			continue
		}
		budgets = append(budgets, int64(gb*1024*1024*1024))
	}
	return budgets
}

// gpuVisibleDevices builds ROCR_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES strings
// from the detected topology, honouring any AIRLLM_GPU_TOPOLOGY override.
// topologyOverride is the raw AIRLLM_GPU_TOPOLOGY env value (may be empty).
func gpuVisibleDevices(topo GPUTopology, topologyOverride string) (rocr string, cuda string) {
	if topologyOverride != "" {
		// Format: "amd:0,nvidia:0,amd:1"  (comma-separated, colon-separated type:index)
		parts := strings.Split(topologyOverride, ",")
		var rocrList, cudaList []string
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			kv := strings.SplitN(p, ":", 2)
			if len(kv) != 2 {
				slog.Warn("AIRLLM_GPU_TOPOLOGY: malformed entry, skipping", "entry", p)
				continue
			}
			kind := strings.TrimSpace(kv[0])
			idxStr := strings.TrimSpace(kv[1])
			idx, err := strconv.Atoi(idxStr)
			if err != nil {
				slog.Warn("AIRLLM_GPU_TOPOLOGY: non-numeric index, skipping", "entry", p)
				continue
			}
			switch strings.ToLower(kind) {
			case "amd", "rocm", "hip":
				rocrList = append(rocrList, strconv.Itoa(idx))
			case "nvidia", "cuda", "nv":
				cudaList = append(cudaList, strconv.Itoa(idx))
			default:
				slog.Warn("AIRLLM_GPU_TOPOLOGY: unknown GPU type, skipping", "type", kind)
			}
		}
		slog.Info("AIRLLM_GPU_TOPOLOGY override applied",
			"ROCR_VISIBLE_DEVICES", strings.Join(rocrList, ","),
			"CUDA_VISIBLE_DEVICES", strings.Join(cudaList, ","))
		return strings.Join(rocrList, ","), strings.Join(cudaList, ",")
	}

	// No override — use all detected devices in discovered order.
	var rocrList, cudaList []string
	for _, d := range topo.Devices {
		switch d.Type {
		case GPUTypeAMD:
			rocrList = append(rocrList, strconv.Itoa(d.Index))
		case GPUTypeNVIDIA:
			cudaList = append(cudaList, strconv.Itoa(d.Index))
		}
	}
	return strings.Join(rocrList, ","), strings.Join(cudaList, ",")
}

type Server struct {
	modelPath  string
	port       int // port the Go proxy listens on (parent ollama talks here)
	pythonPort int // Python airllm_runner.py HTTP server — must differ from port
	pythonBase string
	pythonCmd  *exec.Cmd
	status     llm.ServerStatus
	progress   float32
	mu         sync.Mutex
	commitMu   sync.Mutex // serializes Commit handlers; pairs Add/Done on ready for reloads
	// commitReturnedOnce is true after any Commit handler has finished (success or failure).
	// First Commit pairs with ready.Add(1) in NewServer; subsequent Commits must Add(1) before Done
	// or sync.WaitGroup panics ("negative WaitGroup counter") and the runner dies.
	commitReturnedOnce bool
	ready              sync.WaitGroup
	httpClient         *http.Client
	pythonDone         chan struct{} // closed when the Python subprocess exits

	// Health tracking for circuit breaker pattern.
	// After consecutiveErrorLimit errors, the runner starts failing fast instead of
	// retrying. This prevents hammering a dead runner and allows faster fallback.
	consecutiveErrors     int
	consecutiveErrorLimit int   // defaults to 0; set to 5 in NewServer
	lastError            string
}

type statusResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
}

type completionResponse struct {
	Content            string        `json:"content"`
	Logprobs           []llm.Logprob `json:"logprobs"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration int64         `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       int64         `json:"eval_duration"`
}

// flashAttentionForPy maps ggml flash-attn enum to strings the Python runner accepts.
func flashAttentionForPy(f ml.FlashAttentionType) string {
	switch f {
	case ml.FlashAttentionAuto:
		return "auto"
	case ml.FlashAttentionDisabled:
		return "disabled"
	case ml.FlashAttentionEnabled:
		return "enabled"
	default:
		return "auto"
	}
}

// pythonLoadBody builds JSON for airllm_runner.py /load using snake_case keys and string operation.
func pythonLoadBody(req llm.LoadRequest, modelPath string) ([]byte, error) {
	m := map[string]any{
		"operation":        req.Operation.String(),
		"model_path":       modelPath,
		"lora_path":        req.LoraPath,
		"projector_path":   req.ProjectorPath,
		"parallel":         req.Parallel,
		"batch_size":       req.BatchSize,
		"kv_size":          req.KvSize,
		"kv_cache_type":    req.KvCacheType,
		"flash_attention":  flashAttentionForPy(req.FlashAttention),
		"num_threads":      req.NumThreads,
		"multi_user_cache": req.MultiUserCache,
		"gpu_layers":       req.GPULayers,
		"main_gpu":         req.MainGPU,
		"use_mmap":         req.UseMmap,
	}
	return json.Marshal(m)
}

func optionsToMap(o *api.Options) map[string]interface{} {
	if o == nil {
		return map[string]interface{}{}
	}
	b, err := json.Marshal(o)
	if err != nil {
		return map[string]interface{}{}
	}
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		return map[string]interface{}{}
	}
	return m
}

// pythonCompletionBody builds JSON for airllm_runner.py /completion (snake_case, options object).
func pythonCompletionBody(req llm.CompletionRequest) ([]byte, error) {
	m := map[string]any{
		"prompt":       req.Prompt,
		"images":       req.Media,
		"grammar":      req.Grammar,
		"options":      optionsToMap(req.Options),
		"logprobs":     req.Logprobs,
		"top_logprobs": req.TopLogprobs,
		"shift":        req.Shift,
		"truncate":     req.Truncate,
	}
	return json.Marshal(m)
}

func NewServer(modelPath string, port int) *Server {
	// Python must bind a different port than this Go process; otherwise HTTPClient.Post
	// to /load would hit this same handler with pythonLoadBody JSON and break llm.LoadRequest decode.
	pyPort := port + 1
	if pyPort > 65535 {
		pyPort = port - 1
	}
	s := &Server{
		modelPath:  modelPath,
		port:       port,
		pythonPort: pyPort,
		pythonBase: fmt.Sprintf("http://127.0.0.1:%d", pyPort),
		status:     llm.ServerStatusLaunched,
		httpClient: &http.Client{
			// 10-minute timeout per request — long enough for model loading or
			// a single inference step on large models, but prevents indefinite hangs.
			Timeout: 10 * time.Minute,
		},
		pythonDone:            make(chan struct{}),
		consecutiveErrorLimit: 5, // open circuit breaker after 5 consecutive errors
	}
	s.ready.Add(1)
	return s
}

func (s *Server) startPythonRunner() error {
	pythonRunnerPath := findPythonRunner()
	if pythonRunnerPath == "" {
		return errors.New("airllm_runner.py not found")
	}

	// HC-51: discover heterogeneous GPU topology (AMD ROCm + NVIDIA CUDA) before
	// starting the Python subprocess so we can set ROCR_VISIBLE_DEVICES,
	// CUDA_VISIBLE_DEVICES, and AIRLLM_GPU_TOPOLOGY correctly.
	topo := discoverGPUTopology()
	topoOverride := os.Getenv("AIRLLM_GPU_TOPOLOGY")
	rocrVisible, cudaVisible := gpuVisibleDevices(topo, topoOverride)

	// HC-49: Multi-GPU weight streaming orchestration.
	// Detect number of GPUs to use and per-GPU VRAM budgets.
	gpuCountStr := os.Getenv("AIRLLM_GPU_COUNT")
	gpuCount := 0
	if gpuCountStr != "" {
		if n, err := strconv.Atoi(gpuCountStr); err == nil && n > 0 {
			gpuCount = n
		}
	}

	vramBudgetGB := os.Getenv("AIRLLM_VRAM_BUDGET_GB")
	vramBudgets := parseVRAMBudget(vramBudgetGB)

	// Detect VRAM per GPU via rocm-smi / nvidia-smi for logging.
	detectedVRAM := gpuVRAMBytes()
	var vramSummary []string
	for i, bytes := range detectedVRAM {
		gb := float64(bytes) / 1024 / 1024 / 1024
		budgetStr := ""
		if i < len(vramBudgets) {
			budgetGb := float64(vramBudgets[i]) / 1024 / 1024 / 1024
			budgetStr = fmt.Sprintf(" (budget=%.0f GB)", budgetGb)
		}
		vramSummary = append(vramSummary, fmt.Sprintf("GPU %d: %.1f GB%s", i, gb, budgetStr))
	}
	if len(vramSummary) > 0 {
		slog.Info("GPU VRAM detected", "gpus", vramSummary)
	}

	// HC-50: NUMA-aware NVME placement — discover NUMA topology, map NVMe drives to
	// NUMA nodes by PCIe bus ID, and pass a GPU→NVMe proximity map to the Python
	// runner so shard reads are routed to the NUMA-closest device.
	var numaProximityJSON string
	if nvmeMountsRaw := os.Getenv("AIRLLM_NVME_MOUNTS"); nvmeMountsRaw != "" {
		// Comma-separated list of NVMe mount paths, e.g. /mnt/nvme0,/mnt/nvme1
		nvmeMounts := strings.Split(nvmeMountsRaw, ",")
		for i := range nvmeMounts {
			nvmeMounts[i] = strings.TrimSpace(nvmeMounts[i])
		}
		nvmeToNUMA := mapNVMEToNUMA(nvmeMounts)
		proxMap := buildGPUToNVMEProximityMap(topo, nvmeToNUMA)
		if jsonBytes, err := json.Marshal(proxMap); err == nil {
			numaProximityJSON = string(jsonBytes)
			// Log the proximity map for operator visibility.
			var logLines []string
			for gpuIdx, mounts := range proxMap {
				if len(mounts) > 0 {
					logLines = append(logLines, fmt.Sprintf("GPU %d→%s (NUMA-closest)", gpuIdx, mounts[0]))
				}
			}
			if len(logLines) > 0 {
				slog.Info("HC-50 GPU→NVMe NUMA proximity", "mappings", logLines)
			}
		}
	}

	// Set AIRLLM_MULTI_GPU when GPU count is explicitly specified.
	multiGPU := gpuCount > 1

	cmd := exec.Command("python3", pythonRunnerPath,
		"--model", s.modelPath,
		"--port", strconv.Itoa(s.pythonPort),
	)
	// Inherit full parent env (HIP_VISIBLE_DEVICES, HSA_*, etc.) for GPU visibility in PyTorch.
	// We override the GPU visibility variables based on discovered topology.
	childEnv := append(os.Environ(),
		"AIRLLM_COMPRESSION="+os.Getenv("AIRLLM_COMPRESSION"),
		"PYTHONPATH="+airllmPythonPath(pythonRunnerPath),
		// HC-51: use topology-aware values; parent env is a fallback only.
		"ROCR_VISIBLE_DEVICES="+rocrVisible,
		"CUDA_VISIBLE_DEVICES="+cudaVisible,
	)
	// HC-49: Pass multi-GPU configuration to the Python runner.
	if multiGPU {
		childEnv = append(childEnv, "AIRLLM_MULTI_GPU=1")
		childEnv = append(childEnv, fmt.Sprintf("AIRLLM_GPU_COUNT=%d", gpuCount))
		// Pass per-GPU VRAM budgets as AIRLLM_GPU_VRAM_BUDGET_<N>.
		for i, budget := range vramBudgets {
			childEnv = append(childEnv, fmt.Sprintf("AIRLLM_GPU_VRAM_BUDGET_%d=%d", i, budget))
		}
		slog.Info("HC-49 multi-GPU mode enabled",
			"gpu_count", gpuCount,
			"vram_budget_gb", vramBudgetGB,
			"AIRLLM_MULTI_GPU", "1")
	} else {
		childEnv = append(childEnv, "AIRLLM_MULTI_GPU=0")
	}

	// HC-50: Pass NUMA proximity map as JSON to Python so it can route shard reads
	// to the NUMA-closest NVMe for each GPU.
	if numaProximityJSON != "" {
		childEnv = append(childEnv, "AIRLLM_NUMA_PROXIMITY_MAP="+numaProximityJSON)
	}

	if _, ok := os.LookupEnv("AIRLLM_DEVICE"); !ok {
		// On ROCm, HIP uses device indices that are independent of rocr-visible-devices order.
		// Default to the first visible AMD device when AMD GPUs are detected;
		// otherwise fall back to the first CUDA device. The Python runner probes
		// torch.cuda to confirm the device is actually usable.
		if len(topo.Devices) > 0 && topo.Devices[0].Type == GPUTypeAMD {
			childEnv = append(childEnv, "AIRLLM_DEVICE=hip:0")
		} else {
			childEnv = append(childEnv, "AIRLLM_DEVICE=cuda:0")
		}
	}
	// Pass the GPU topology string through to the Python runner so it can log it
	// and optionally re-order its internal device list.
	if topoOverride != "" {
		childEnv = append(childEnv, "AIRLLM_GPU_TOPOLOGY="+topoOverride)
	} else if len(topo.Devices) > 0 {
		// Encode the discovered topology as a canonical string for the Python runner.
		var parts []string
		for _, d := range topo.Devices {
			parts = append(parts, fmt.Sprintf("%s:%d", d.Type, d.Index))
		}
		childEnv = append(childEnv, "AIRLLM_GPU_TOPOLOGY="+strings.Join(parts, ","))
	}

	cmd.Env = childEnv
	adev := os.Getenv("AIRLLM_DEVICE")
	if adev == "" {
		adev = "cuda:0"
	}
	slog.Info("airllm python env (subset)",
		"AIRLLM_DEVICE", adev,
		"HIP_VISIBLE_DEVICES", os.Getenv("HIP_VISIBLE_DEVICES"),
		"ROCR_VISIBLE_DEVICES", rocrVisible,
		"CUDA_VISIBLE_DEVICES", cudaVisible,
		"AIRLLM_GPU_TOPOLOGY", os.Getenv("AIRLLM_GPU_TOPOLOGY"),
		"gpu_devices", len(topo.Devices))
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	s.pythonCmd = cmd
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start Python runner: %w", err)
	}

	go func() {
		cmd.Wait()
		slog.Info("Python runner exited")
		close(s.pythonDone) // signal that the Python process has exited
	}()

	return s.waitForReady()
}

func (s *Server) waitForReady() error {
	for i := 0; i < 60; i++ {
		resp, err := s.httpClient.Get(s.pythonBase + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				slog.Info("Python runner is ready")
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return errors.New("timeout waiting for Python runner to start")
}

// airllmPythonPath sets import path for the `airllm` package: packaged install
// or a git checkout (src/airllm/air_llm next to the repo root).
// Also includes src/prismalama for prismalama-specific integration code (HC-52 nvme_striping).
func airllmPythonPath(pythonRunnerPath string) string {
	const packaged = "/usr/share/ollama/airllm:/usr/share/ollama/airllm/air_llm"
	runnerDir := filepath.Dir(pythonRunnerPath)
	// .../runner/airllmrunner/airllm_runner.py -> .../src/airllm/air_llm
	devAirLLM := filepath.Join(runnerDir, "..", "..", "src", "airllm", "air_llm")
	// prismalama-specific integration modules (HC-52 nvme_striping, etc.)
	devPrismalama := filepath.Join(runnerDir, "..", "..", "src", "prismalama")

	var parts []string
	if st, err := os.Stat(filepath.Join(devAirLLM, "airllm")); err == nil && st.IsDir() {
		parts = append(parts, devAirLLM)
	}
	if st, err := os.Stat(devPrismalama); err == nil && st.IsDir() {
		parts = append(parts, devPrismalama)
	}
	parts = append(parts, packaged)

	extra := os.Getenv("PRISMALAMA_AIRLLM_PYTHONPATH")
	if extra != "" {
		parts = append([]string{extra}, parts...)
	}
	return strings.Join(parts, ":")
}

func findPythonRunner() string {
	exeDir := filepath.Dir(os.Args[0])
	candidates := []string{
		"/usr/share/ollama/airllm_runner.py",
		"/usr/lib/ollama/airllm_runner.py",
		filepath.Join(exeDir, "airllm_runner.py"),
		filepath.Join(exeDir, "runner", "airllmrunner", "airllm_runner.py"),
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	var req llm.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		slog.Error("airllm load JSON decode", "error", err)
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	switch req.Operation {
	case llm.LoadOperationCommit:
		// WaitGroup: NewServer adds 1. First Commit does one Done (pairs with NewServer).
		// Any later Commit must Add(1) before Done or WaitGroup panics and the runner exits.
		s.commitMu.Lock()
		// commitAdded tracks whether Add(1) was called in THIS invocation so that
		// early-returns (before the defer runs) don't cause a WaitGroup underflow.
		commitAdded := s.commitReturnedOnce
		if s.commitReturnedOnce {
			s.ready.Add(1)
		}
		defer func() {
			if commitAdded {
				s.ready.Done()
			}
			s.commitReturnedOnce = true
			s.commitMu.Unlock()
		}()

		s.mu.Lock()
		s.status = llm.ServerStatusLoadingModel
		s.mu.Unlock()

		if err := s.startPythonRunner(); err != nil {
			slog.Error("failed to start Python runner", "error", err)
			// commitAdded=false here: Add(1) was not called above, so Don't call Done().
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}

		proxyReq, err := pythonLoadBody(req, s.modelPath)
		if err != nil {
			slog.Error("airllm load proxy body", "error", err)
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}
		resp, err := s.httpClient.Post(s.pythonBase+"/load", "application/json", bytes.NewReader(proxyReq))
		if err != nil {
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}
		if resp.StatusCode >= 400 {
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: strings.TrimSpace(string(body))})
			return
		}

		var pyResp llm.LoadResponse
		if err := json.Unmarshal(body, &pyResp); err != nil {
			json.NewEncoder(w).Encode(llm.LoadResponse{Success: false, Error: err.Error()})
			return
		}

		if pyResp.Success {
			s.mu.Lock()
			s.status = llm.ServerStatusReady
			s.mu.Unlock()
		}

		json.NewEncoder(w).Encode(pyResp)

	case llm.LoadOperationClose:
		if s.pythonCmd != nil && s.pythonCmd.Process != nil {
			s.pythonCmd.Process.Kill()
		}
		// Reset pythonDone so a new runner can be started on a subsequent Commit.
		s.pythonDone = make(chan struct{})
		json.NewEncoder(w).Encode(llm.LoadResponse{Success: true})

	default:
		json.NewEncoder(w).Encode(llm.LoadResponse{Success: true})
	}
}

func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	s.ready.Wait()

	var req llm.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		slog.Error("airllm completion JSON decode", "error", err)
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	proxyReq, err := pythonCompletionBody(req)
	if err != nil {
		slog.Error("airllm completion proxy body", "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp, err := s.httpClient.Post(s.pythonBase+"/completion", "application/json", bytes.NewReader(proxyReq))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	decoder := json.NewDecoder(resp.Body)
	for {
		var pyResp completionResponse
		if err := decoder.Decode(&pyResp); err != nil {
			break
		}

		var doneReason llm.DoneReason
		switch pyResp.DoneReason {
		case "stop":
			doneReason = llm.DoneReasonStop
		case "length":
			doneReason = llm.DoneReasonLength
		default:
			doneReason = llm.DoneReasonStop
		}

		ollamaResp := llm.CompletionResponse{
			Content:            pyResp.Content,
			Logprobs:           pyResp.Logprobs,
			Done:               pyResp.Done,
			DoneReason:         doneReason,
			PromptEvalCount:    pyResp.PromptEvalCount,
			PromptEvalDuration: time.Duration(pyResp.PromptEvalDuration),
			EvalCount:          pyResp.EvalCount,
			EvalDuration:       time.Duration(pyResp.EvalDuration),
		}

		if err := json.NewEncoder(w).Encode(&ollamaResp); err != nil {
			return
		}
		flusher.Flush()

		if pyResp.Done {
			break
		}
	}
}

// healthResponse extends llm.ServerStatusResponse with additional diagnostic fields.
type healthResponse struct {
	Status             llm.ServerStatus `json:"status"`
	Progress           float32          `json:"progress"`
	ModelPath          string           `json:"model_path"`
	PythonAlive        bool             `json:"python_alive"`
	PythonPID          int              `json:"python_pid,omitempty"`
	ConsecutiveErrors  int              `json:"consecutive_errors"`
	LastError          string           `json:"last_error,omitempty"`
	ReadyForInference  bool             `json:"ready_for_inference"`
}

// recordError increments the consecutive error counter.
// When consecutiveErrorLimit is reached, the runner becomes unavailable.
func (s *Server) recordError(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.consecutiveErrors++
	s.lastError = err.Error()
	if s.consecutiveErrors >= s.consecutiveErrorLimit {
		s.status = llm.ServerStatusError
	}
}

// resetErrors clears the consecutive error counter on successful operations.
func (s *Server) resetErrors() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.consecutiveErrors = 0
	s.lastError = ""
}

// readyz returns 200 OK when the runner is ready to accept inference requests,
// or 503 Service Unavailable when still loading or errored.
// This is useful for Kubernetes readiness probes.
func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	status := s.status
	s.mu.Unlock()

	// ready.Wait() blocks until the first Commit finishes (model loaded).
	// Use status as a quick check; if not ready, use WaitGroup to wait.
	if status != llm.ServerStatusReady {
		http.Error(w, fmt.Sprintf("runner not ready: status=%d", status), http.StatusServiceUnavailable)
		return
	}

	// Double-check by trying WaitGroup with a short timeout.
	done := make(chan struct{})
	go func() {
		s.ready.Wait()
		close(done)
	}()

	select {
	case <-done:
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `{"ready":true}`)
	case <-time.After(5 * time.Second):
		http.Error(w, "timeout waiting for runner to be ready", http.StatusServiceUnavailable)
	}
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.Lock()
	status := s.status
	progress := s.progress
	modelPath := s.modelPath
	consecutiveErrors := s.consecutiveErrors
	lastError := s.lastError
	var pythonPID int
	pythonAlive := false
	if s.pythonCmd != nil && s.pythonCmd.Process != nil {
		pythonPID = s.pythonCmd.Process.Pid
		pythonAlive = s.pythonCmd.Process.Pid > 0
	}
	s.mu.Unlock()

	resp := healthResponse{
		Status:            status,
		Progress:          progress,
		ModelPath:         modelPath,
		PythonAlive:       pythonAlive,
		PythonPID:         pythonPID,
		ConsecutiveErrors: consecutiveErrors,
		LastError:         lastError,
		ReadyForInference: status == llm.ServerStatusReady && consecutiveErrors < 5,
	}

	json.NewEncoder(w).Encode(&resp)
}

func Execute(args []string) error {
	fs := flag.NewFlagSet("airllm-runner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model")
	port := fs.Int("port", 8080, "Port to listen on")
	_ = fs.Bool("verbose", false, "Verbose output")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "AirLLM Runner for Ollama\n")
		fs.PrintDefaults()
	}

	if err := fs.Parse(args); err != nil {
		return err
	}

	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("starting AirLLM runner")

	server := NewServer(*mpath, *port)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen error: %w", err)
	}
	defer listener.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /load", server.load)
	mux.HandleFunc("/completion", server.completion)
	mux.HandleFunc("/health", server.health)
	mux.HandleFunc("/readyz", server.readyz) // returns 200 only when ready to serve

	httpServer := &http.Server{
		Handler: mux,
	}

	log.Println("AirLLM Runner listening on", addr)
	return httpServer.Serve(listener)
}
