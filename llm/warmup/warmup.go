package warmup

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

type WarmupStrategy int

const (
	NoWarmup WarmupStrategy = iota
	QuickWarmup
	FullWarmup
	ProfileBased
)

type GraphKey struct {
	ModelHash uint64
	BatchSize int
	SeqLen    int
	DeviceID  ml.DeviceID
}

type CompiledGraph struct {
	key       GraphKey
	handle    interface{}
	createdAt time.Time
}

type AOTCompilation struct {
	batchSizes      []int
	sequenceLengths []int
	compiledGraphs  map[GraphKey]*CompiledGraph
	cacheDir        string
	mu              sync.RWMutex
	model           *ml.Model
}

type Model interface {
	NumLayers() int
	Forward(ctx ml.Context, input ml.Tensor) (ml.Tensor, error)
}

func NewAOTCompilation(cacheDir string) *AOTCompilation {
	return &AOTCompilation{
		compiledGraphs:  make(map[GraphKey]*CompiledGraph),
		cacheDir:        cacheDir,
		batchSizes:      []int{1, 2, 4, 8, 16, 32},
		sequenceLengths: []int{128, 256, 512, 1024, 2048},
	}
}

func (a *AOTCompilation) SetBatchSizes(sizes []int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.batchSizes = sizes
}

func (a *AOTCompilation) SetSequenceLengths(lengths []int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.sequenceLengths = lengths
}

func (a *AOTCompilation) Compile(ctx context.Context, model Model, deviceID ml.DeviceID) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	modelHash := computeModelHash(model)

	shapes := make([]GraphKey, 0, len(a.batchSizes)*len(a.sequenceLengths))
	for _, batchSize := range a.batchSizes {
		for _, seqLen := range a.sequenceLengths {
			shapes = append(shapes, GraphKey{
				ModelHash: modelHash,
				BatchSize: batchSize,
				SeqLen:    seqLen,
				DeviceID:  deviceID,
			})
		}
	}

	for _, shape := range shapes {
		if _, ok := a.compiledGraphs[shape]; !ok {
			graph := &CompiledGraph{
				key:       shape,
				handle:    nil,
				createdAt: time.Now(),
			}
			a.compiledGraphs[shape] = graph
		}
	}

	return a.persistCache()
}

func (a *AOTCompilation) GetCompiledGraph(key GraphKey) (*CompiledGraph, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	graph, ok := a.compiledGraphs[key]
	return graph, ok
}

func (a *AOTCompilation) persistCache() error {
	if a.cacheDir == "" {
		return nil
	}

	if err := os.MkdirAll(a.cacheDir, 0755); err != nil {
		return err
	}

	return nil
}

func (a *AOTCompilation) LoadCache() error {
	if a.cacheDir == "" {
		return nil
	}

	return nil
}

func computeModelHash(model Model) uint64 {
	hash := sha256.New()
	hash.Write([]byte("model"))
	return 0
}

type WarmupRunner struct {
	strategy    WarmupStrategy
	compilation *AOTCompilation
	model       Model
	backend     ml.Backend
	profileData *ProfileData
	mu          sync.RWMutex
}

type ProfileData struct {
	BatchSizes      []int
	SequenceLengths []int
	AccessCount     map[GraphKey]int
	LastAccess      map[GraphKey]time.Time
}

func NewWarmupRunner(strategy WarmupStrategy, model Model, backend ml.Backend) *WarmupRunner {
	return &WarmupRunner{
		strategy:    strategy,
		model:       model,
		backend:     backend,
		compilation: NewAOTCompilation(""),
		profileData: &ProfileData{
			AccessCount: make(map[GraphKey]int),
			LastAccess:  make(map[GraphKey]time.Time),
		},
	}
}

func (w *WarmupRunner) Run(ctx context.Context) error {
	switch w.strategy {
	case NoWarmup:
		return nil

	case QuickWarmup:
		return w.quickWarmup(ctx)

	case FullWarmup:
		return w.fullWarmup(ctx)

	case ProfileBased:
		return w.profileBasedWarmup(ctx)

	default:
		return w.quickWarmup(ctx)
	}
}

func (w *WarmupRunner) quickWarmup(ctx context.Context) error {
	_ = ctx
	return nil
}

func (w *WarmupRunner) fullWarmup(ctx context.Context) error {
	_ = ctx
	return nil
}

func (w *WarmupRunner) profileBasedWarmup(ctx context.Context) error {
	_ = ctx
	return nil
}

func (w *WarmupRunner) RecordAccess(batchSize, seqLen int) {
	w.mu.Lock()
	defer w.mu.Unlock()

	key := GraphKey{
		BatchSize: batchSize,
		SeqLen:    seqLen,
	}

	w.profileData.AccessCount[key]++
	w.profileData.LastAccess[key] = time.Now()
}

func (w *WarmupRunner) getHotShapes() []GraphKey {
	w.mu.RLock()
	defer w.mu.RUnlock()

	type shapeAccess struct {
		key       GraphKey
		accessCnt int
	}

	shapes := make([]shapeAccess, 0, len(w.profileData.AccessCount))
	for key, cnt := range w.profileData.AccessCount {
		shapes = append(shapes, shapeAccess{key: key, accessCnt: cnt})
	}

	for i := 0; i < len(shapes)-1; i++ {
		for j := i + 1; j < len(shapes); j++ {
			if shapes[j].accessCnt > shapes[i].accessCnt {
				shapes[i], shapes[j] = shapes[j], shapes[i]
			}
		}
	}

	result := make([]GraphKey, 0, min(10, len(shapes)))
	for i := 0; i < min(10, len(shapes)); i++ {
		result = append(result, shapes[i].key)
	}

	return result
}

func (w *WarmupRunner) SetCacheDir(dir string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.compilation.cacheDir = dir
}

func (w *WarmupRunner) GetCacheDir() string {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.compilation.cacheDir
}

type JITCompiler struct {
	backend     ml.Backend
	compilePool *sync.Pool
	mu          sync.RWMutex
	jitCache    map[string]interface{}
}

func NewJITCompiler(backend ml.Backend) *JITCompiler {
	return &JITCompiler{
		backend:  backend,
		jitCache: make(map[string]interface{}),
		compilePool: &sync.Pool{
			New: func() interface{} {
				return nil
			},
		},
	}
}

func (j *JITCompiler) CompileShape(ctx context.Context, batchSize, seqLen int) (interface{}, error) {
	key := computeShapeKey(batchSize, seqLen)

	j.mu.RLock()
	if handle, ok := j.jitCache[key]; ok {
		j.mu.RUnlock()
		return handle, nil
	}
	j.mu.RUnlock()

	_ = ctx

	handle := j.compilePool.Get()
	j.mu.Lock()
	j.jitCache[key] = handle
	j.mu.Unlock()

	return handle, nil
}

func (j *JITCompiler) ClearCache() {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.jitCache = make(map[string]interface{})
}

func computeShapeKey(batchSize, seqLen int) string {
	hash := sha256.New()
	hash.Write([]byte{byte(batchSize), byte(batchSize >> 8)})
	hash.Write([]byte{byte(seqLen), byte(seqLen >> 8)})
	return hex.EncodeToString(hash.Sum(nil))
}

type WarmupProfile struct {
	ModelPath     string
	WarmupHistory []WarmupRun
}

type WarmupRun struct {
	Timestamp  time.Time
	Strategy   WarmupStrategy
	Duration   time.Duration
	BatchSizes []int
	SeqLengths []int
}

func SaveWarmupProfile(path string, profile *WarmupProfile) error {
	data, err := os.Create(path)
	if err != nil {
		return err
	}
	defer data.Close()
	_ = data
	return nil
}

func LoadWarmupProfile(path string) (*WarmupProfile, error) {
	_, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	return nil, nil
}
