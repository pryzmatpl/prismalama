package warmup

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
)

func TestWarmupStrategy(t *testing.T) {
	if NoWarmup != 0 {
		t.Errorf("expected NoWarmup = 0, got %d", NoWarmup)
	}
	if QuickWarmup != 1 {
		t.Errorf("expected QuickWarmup = 1, got %d", QuickWarmup)
	}
	if FullWarmup != 2 {
		t.Errorf("expected FullWarmup = 2, got %d", FullWarmup)
	}
	if ProfileBased != 3 {
		t.Errorf("expected ProfileBased = 3, got %d", ProfileBased)
	}
}

func TestGraphKey(t *testing.T) {
	key := GraphKey{
		ModelHash: 12345,
		BatchSize: 4,
		SeqLen:    1024,
		DeviceID:  ml.DeviceID{ID: "0"},
	}

	if key.ModelHash != 12345 {
		t.Errorf("expected ModelHash 12345, got %d", key.ModelHash)
	}
	if key.BatchSize != 4 {
		t.Errorf("expected BatchSize 4, got %d", key.BatchSize)
	}
	if key.SeqLen != 1024 {
		t.Errorf("expected SeqLen 1024, got %d", key.SeqLen)
	}
}

func TestCompiledGraph(t *testing.T) {
	graph := &CompiledGraph{
		key:       GraphKey{BatchSize: 4, SeqLen: 1024},
		handle:    nil,
		createdAt: time.Now(),
	}

	if graph.key.BatchSize != 4 {
		t.Errorf("expected BatchSize 4, got %d", graph.key.BatchSize)
	}
}

func TestAOTCompilation_New(t *testing.T) {
	comp := NewAOTCompilation("/cache/dir")
	if comp == nil {
		t.Fatal("expected non-nil AOTCompilation")
	}
	if comp.cacheDir != "/cache/dir" {
		t.Errorf("expected /cache/dir, got %s", comp.cacheDir)
	}
	if len(comp.batchSizes) != 6 {
		t.Errorf("expected 6 default batch sizes, got %d", len(comp.batchSizes))
	}
}

func TestAOTCompilation_SetBatchSizes(t *testing.T) {
	comp := NewAOTCompilation("")
	comp.SetBatchSizes([]int{1, 2, 4})

	if len(comp.batchSizes) != 3 {
		t.Errorf("expected 3 batch sizes, got %d", len(comp.batchSizes))
	}
	if comp.batchSizes[0] != 1 {
		t.Errorf("expected 1, got %d", comp.batchSizes[0])
	}
}

func TestAOTCompilation_SetSequenceLengths(t *testing.T) {
	comp := NewAOTCompilation("")
	comp.SetSequenceLengths([]int{256, 512, 1024})

	if len(comp.sequenceLengths) != 3 {
		t.Errorf("expected 3 sequence lengths, got %d", len(comp.sequenceLengths))
	}
	if comp.sequenceLengths[0] != 256 {
		t.Errorf("expected 256, got %d", comp.sequenceLengths[0])
	}
}

func TestAOTCompilation_GetCompiledGraph_NotFound(t *testing.T) {
	comp := NewAOTCompilation("")

	key := GraphKey{
		ModelHash: 12345,
		BatchSize: 4,
		SeqLen:    1024,
		DeviceID:  ml.DeviceID{ID: "0"},
	}

	_, ok := comp.GetCompiledGraph(key)
	if ok {
		t.Error("expected not to find non-compiled graph")
	}
}

func TestAOTCompilation_persistCache_EmptyDir(t *testing.T) {
	comp := NewAOTCompilation("")
	err := comp.persistCache()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestAOTCompilation_LoadCache_EmptyDir(t *testing.T) {
	comp := NewAOTCompilation("")
	err := comp.LoadCache()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestProfileData(t *testing.T) {
	profile := &ProfileData{
		BatchSizes:      []int{1, 2, 4},
		SequenceLengths: []int{256, 512},
		AccessCount:     make(map[GraphKey]int),
		LastAccess:      make(map[GraphKey]time.Time),
	}

	if len(profile.BatchSizes) != 3 {
		t.Errorf("expected 3 batch sizes, got %d", len(profile.BatchSizes))
	}
}

func TestWarmupRunner_New(t *testing.T) {
	runner := NewWarmupRunner(QuickWarmup, nil, nil)
	if runner == nil {
		t.Fatal("expected non-nil WarmupRunner")
	}
	if runner.strategy != QuickWarmup {
		t.Errorf("expected QuickWarmup strategy, got %d", runner.strategy)
	}
}

func TestWarmupRunner_Run_NoWarmup(t *testing.T) {
	runner := NewWarmupRunner(NoWarmup, nil, nil)
	err := runner.Run(context.Background())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestWarmupRunner_Run_QuickWarmup(t *testing.T) {
	runner := NewWarmupRunner(QuickWarmup, nil, nil)
	err := runner.Run(context.Background())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestWarmupRunner_Run_FullWarmup(t *testing.T) {
	runner := NewWarmupRunner(FullWarmup, nil, nil)
	err := runner.Run(context.Background())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestWarmupRunner_Run_ProfileBased(t *testing.T) {
	runner := NewWarmupRunner(ProfileBased, nil, nil)
	err := runner.Run(context.Background())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestWarmupRunner_RecordAccess(t *testing.T) {
	runner := NewWarmupRunner(QuickWarmup, nil, nil)
	runner.RecordAccess(4, 1024)
	runner.RecordAccess(4, 1024)
	runner.RecordAccess(8, 512)
}

func TestWarmupRunner_SetCacheDir(t *testing.T) {
	runner := NewWarmupRunner(QuickWarmup, nil, nil)
	runner.SetCacheDir("/test/cache")

	if runner.GetCacheDir() != "/test/cache" {
		t.Errorf("expected /test/cache, got %s", runner.GetCacheDir())
	}
}

func TestJITCompiler_New(t *testing.T) {
	compiler := NewJITCompiler(nil)
	if compiler == nil {
		t.Fatal("expected non-nil JITCompiler")
	}
	if compiler.jitCache == nil {
		t.Error("expected non-nil jitCache")
	}
}

func TestJITCompiler_CompileShape(t *testing.T) {
	compiler := NewJITCompiler(nil)

	handle, err := compiler.CompileShape(context.Background(), 4, 1024)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	_ = handle
}

func TestJITCompiler_CompileShape_Cached(t *testing.T) {
	compiler := NewJITCompiler(nil)

	compiler.CompileShape(context.Background(), 4, 1024)
	handle, err := compiler.CompileShape(context.Background(), 4, 1024)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	_ = handle
}

func TestJITCompiler_ClearCache(t *testing.T) {
	compiler := NewJITCompiler(nil)

	compiler.CompileShape(context.Background(), 4, 1024)
	compiler.ClearCache()

	if len(compiler.jitCache) != 0 {
		t.Error("expected cache to be cleared")
	}
}

func TestWarmupProfile(t *testing.T) {
	profile := &WarmupProfile{
		ModelPath: "/models/llama",
		WarmupHistory: []WarmupRun{
			{
				Timestamp:  time.Now(),
				Strategy:   QuickWarmup,
				Duration:   100 * time.Millisecond,
				BatchSizes: []int{1, 2, 4},
				SeqLengths: []int{256, 512},
			},
		},
	}

	if profile.ModelPath != "/models/llama" {
		t.Errorf("expected /models/llama, got %s", profile.ModelPath)
	}
	if len(profile.WarmupHistory) != 1 {
		t.Errorf("expected 1 warmup run, got %d", len(profile.WarmupHistory))
	}
}

func TestWarmupRun(t *testing.T) {
	run := WarmupRun{
		Timestamp:  time.Now(),
		Strategy:   FullWarmup,
		Duration:   500 * time.Millisecond,
		BatchSizes: []int{1, 2, 4, 8},
		SeqLengths: []int{128, 256, 512, 1024},
	}

	if run.Strategy != FullWarmup {
		t.Errorf("expected FullWarmup, got %d", run.Strategy)
	}
	if run.Duration != 500*time.Millisecond {
		t.Errorf("expected 500ms, got %v", run.Duration)
	}
}

func TestSaveWarmupProfile_EmptyPath(t *testing.T) {
	profile := &WarmupProfile{
		ModelPath: "/models/llama",
	}

	err := SaveWarmupProfile("", profile)
	if err == nil {
		t.Error("expected error for empty path")
	}
}

func TestLoadWarmupProfile_NotFound(t *testing.T) {
	profile, err := LoadWarmupProfile("/nonexistent/profile.json")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
	if profile != nil {
		t.Error("expected nil profile")
	}
}
