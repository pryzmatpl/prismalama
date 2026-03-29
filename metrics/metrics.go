// Package metrics provides Prometheus metrics for the Ollama server.
// Metrics are registered on startup and exposed via the /debug/metrics HTTP endpoint
// when the metrics server is enabled.
package metrics

import (
	"context"
	"expvar"
	"net"
	"net/http"
	"runtime"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const namespace = "ollama"

var (
	// Inference metrics
	InferenceDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "inference",
			Name:      "duration_seconds",
			Help:      "Total time spent processing an inference request (prompt_eval + generation)",
			Buckets:   []float64{0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300},
		},
		[]string{"model", "status"}, // status: "success" | "error"
	)

	PromptEvalDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "inference",
			Name:      "prompt_eval_duration_seconds",
			Help:      "Time spent evaluating the prompt tokens",
			Buckets:   []float64{0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60},
		},
		[]string{"model"},
	)

	GenerationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "inference",
			Name:      "generation_duration_seconds",
			Help:      "Time spent generating output tokens",
			Buckets:   []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30},
		},
		[]string{"model"},
	)

	TokensGenerated = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "inference",
			Name:      "tokens_generated",
			Help:      "Number of tokens generated per request",
			Buckets:   []float64{1, 10, 50, 100, 250, 500, 1000, 2000, 4000},
		},
		[]string{"model"},
	)

	TokensPerSecond = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "inference",
			Name:      "tokens_per_second",
			Help:      "Current inference throughput (tokens/second) for the last completed request",
		},
		[]string{"model"},
	)

	// Request metrics
	ActiveRequests = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "active_requests",
			Help:      "Number of inference requests currently being processed",
		},
	)

	RequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "requests_total",
			Help:      "Total number of inference requests processed",
		},
		[]string{"model", "status"}, // status: "success" | "error" | "cancelled"
	)

	RequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "request_duration_seconds",
			Help:      "End-to-end HTTP request duration including network overhead",
			Buckets:   []float64{0.1, 0.5, 1, 2, 5, 10, 30, 60, 120},
		},
		[]string{"model", "endpoint"}, // endpoint: "generate" | "chat" | "embeddings"
	)

	// Model loading metrics
	ModelLoadDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "model",
			Name:      "load_duration_seconds",
			Help:      "Time taken to fully load a model into GPU and CPU memory",
			Buckets:   []float64{1, 5, 10, 30, 60, 120, 300, 600},
		},
		[]string{"model"},
	)

	ModelLoaded = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "model",
			Name:      "loaded",
			Help:      "Whether a model is currently loaded (1) or not (0)",
		},
		[]string{"model"},
	)

	// VRAM metrics
	VRAMUsedBytes = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "vram",
			Name:      "used_bytes",
			Help:      "Estimated VRAM currently in use per GPU",
		},
		[]string{"device", "model"},
	)

	VRAMTotalBytes = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "vram",
			Name:      "total_bytes",
			Help:      "Total VRAM on the GPU device",
		},
		[]string{"device"},
	)

	VRAMUsagePercent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "vram",
			Name:      "usage_percent",
			Help:      "VRAM utilization as a percentage of total",
		},
		[]string{"device", "model"},
	)

	// Layer offload metrics
	GPULayersOffloaded = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "gpu",
			Name:      "layers_offloaded",
			Help:      "Number of model layers currently offloaded to each GPU",
		},
		[]string{"device", "model"},
	)

	// Cache metrics
	ContextCacheSize = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "cache",
			Name:      "context_size_tokens",
			Help:      "Current KV cache size in tokens",
		},
	)

	// Runner metrics
	RunnerCount = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "runner",
			Name:      "count",
			Help:      "Number of active runner processes",
		},
	)

	RunnerStatus = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "runner",
			Name:      "status",
			Help:      "Runner status: 1=launched, 2=loading, 3=ready, 4=error, 5=unloaded",
		},
		[]string{"model"},
	)
)

// RecordInference records metrics for a completed inference request.
func RecordInference(model string, status string, duration time.Duration, promptEvalDur, genDur time.Duration, tokensGenerated int) {
	InferenceDuration.WithLabelValues(model, status).Observe(duration.Seconds())
	RequestsTotal.WithLabelValues(model, status).Inc()
	if tokensGenerated > 0 && genDur > 0 {
		tps := float64(tokensGenerated) / genDur.Seconds()
		TokensPerSecond.WithLabelValues(model).Set(tps)
	}
	if promptEvalDur > 0 {
		PromptEvalDuration.WithLabelValues(model).Observe(promptEvalDur.Seconds())
	}
	if genDur > 0 {
		GenerationDuration.WithLabelValues(model).Observe(genDur.Seconds())
	}
	if tokensGenerated > 0 {
		TokensGenerated.WithLabelValues(model).Observe(float64(tokensGenerated))
	}
}

// RecordRequestStart marks that a new inference request has started.
func RecordRequestStart() {
	ActiveRequests.Inc()
}

// RecordModelLoad records model loading duration.
func RecordModelLoad(model string, duration time.Duration) {
	ModelLoadDuration.WithLabelValues(model).Observe(duration.Seconds())
	ModelLoaded.WithLabelValues(model).Set(1)
}

// RecordModelUnload marks a model as no longer loaded.
func RecordModelUnload(model string) {
	ModelLoaded.WithLabelValues(model).Set(0)
}

// RecordVRAMUpdate updates VRAM usage metrics.
func RecordVRAMUpdate(device, model string, usedBytes, totalBytes uint64) {
	VRAMUsedBytes.WithLabelValues(device, model).Set(float64(usedBytes))
	if totalBytes > 0 {
		VRAMTotalBytes.WithLabelValues(device).Set(float64(totalBytes))
		VRAMUsagePercent.WithLabelValues(device, model).Set(float64(usedBytes) / float64(totalBytes) * 100)
	}
}

// Handler returns an HTTP handler that exposes Prometheus metrics.
// This can be registered on /debug/metrics in addition to pprof endpoints.
func Handler() http.Handler {
	return promhttp.Handler()
}

// HandlerWithExpvar returns a combined handler that exposes both Prometheus
// metrics and Go expvar variables (including memory stats, goroutine counts, etc.)
func HandlerWithExpvar() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/debug/metrics", promhttp.Handler())
	mux.Handle("/debug/vars", expvar.Handler())
	return mux
}

// MetricsServer is a lightweight HTTP server that exposes metrics on a separate port.
type MetricsServer struct {
	ln   net.Listener
	srv  *http.Server
	done chan struct{}
}

// NewMetricsServer creates a metrics server listening on the given address.
// The server exposes Prometheus metrics at /debug/metrics and Go expvar at /debug/vars.
func NewMetricsServer(addr string) (*MetricsServer, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}

	mux := http.NewServeMux()
	mux.Handle("/debug/metrics", promhttp.Handler())
	mux.Handle("/debug/vars", expvar.Handler())

	srv := &http.Server{Handler: mux}
	ms := &MetricsServer{ln: ln, srv: srv, done: make(chan struct{})}
	go srv.Serve(ln)
	return ms, nil
}

// Close stops the metrics server.
func (ms *MetricsServer) Close() error {
	select {
	case <-ms.done:
		return nil
	default:
	}
	close(ms.done)
	return ms.srv.Shutdown(context.Background())
}

func init() {
	// Export key Go runtime stats via expvar.
	// These complement Prometheus metrics by providing low-overhead continuous stats.
	expvar.Publish("go_memstats_alloc_bytes", expvar.Func(func() any {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return int64(m.Alloc)
	}))
	expvar.Publish("go_goroutines", expvar.Func(func() any {
		return runtime.NumGoroutine()
	}))
	expvar.Publish("go_threads", expvar.Func(func() any {
		return runtime.NumCPU()
	}))
}
