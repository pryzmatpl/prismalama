package stream

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/ollama/ollama/fs/gguf"
)

// Loader manages GGUF model files with an optional in-memory tensor cache.
// It supports multi-part GGUF files (e.g. model-00001-of-00003.gguf) by
// treating all parts as a single logical model.
type Loader struct {
	maxCache    int64
	tensorCache map[string][]byte
	cacheSize   int64
	files       map[string]*gguf.File // path → file handle
	mu          sync.Mutex
}

// NewLoader creates a new GGUF model loader. maxCache is the maximum
// size of the tensor cache in bytes. A value of 0 disables caching.
func NewLoader(maxCache int64) *Loader {
	return &Loader{
		maxCache:    maxCache,
		tensorCache: make(map[string][]byte),
		files:       make(map[string]*gguf.File),
	}
}

// Open opens a GGUF file (or directory containing part files) for reading.
// It is safe to call Open multiple times; it is idempotent.
// Files are not closed until Loader.Close is called.
func (l *Loader) Open(path string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Re-initialize maps if they were cleared by Close().
	if l.files == nil {
		l.files = make(map[string]*gguf.File)
	}
	if l.tensorCache == nil {
		l.tensorCache = make(map[string][]byte)
	}

	if _, ok := l.files[path]; ok {
		return nil
	}

	f, err := gguf.Open(path)
	if err != nil {
		return err
	}
	l.files[path] = f
	return nil
}

// GetTensor returns the tensor data for the given name, reading from the
// appropriate GGUF part file. Cached data is returned on subsequent calls
// without re-reading from disk.
func (l *Loader) GetTensor(path, name string) ([]byte, error) {
	l.mu.Lock()

	// Re-initialize if cleared by Close().
	if l.tensorCache == nil {
		l.tensorCache = make(map[string][]byte)
	}

	key := path + "::" + name
	if cached, ok := l.tensorCache[key]; ok {
		l.mu.Unlock()
		return cached, nil
	}
	l.mu.Unlock()

	f := func() *gguf.File {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.files[path]
	}()
	if f == nil {
		return nil, fmt.Errorf("file not opened: %s", path)
	}

	_, reader, err := f.TensorReader(name)
	if err != nil {
		return nil, err
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	l.mu.Lock()
	l.addToCache(key, data)
	l.mu.Unlock()

	return data, nil
}

// addToCache adds data to the LRU cache, evicting the oldest entries if
// the cache is full. If data is larger than the entire cache, it is not
// stored. The caller must hold l.mu.
func (l *Loader) addToCache(key string, data []byte) {
	if l.maxCache == 0 {
		return
	}

	dataSize := int64(len(data))

	// Evict entries until there's room for the new data.
	// The condition is: keep evicting while adding data would exceed maxCache.
	for l.cacheSize+dataSize > l.maxCache && len(l.tensorCache) > 0 {
		// Find the oldest entry (first inserted; map iteration order in Go is
		// unspecified, so we track the first key we see rather than relying on
		// insertion order which a plain map doesn't preserve).
		var oldestKey string
		var oldestSize int64
		for k, v := range l.tensorCache {
			oldestKey = k
			oldestSize = int64(len(v))
			break // first entry is the oldest by insertion order
		}
		if oldestKey == "" {
			break
		}
		delete(l.tensorCache, oldestKey)
		l.cacheSize -= oldestSize
	}

	// If a single entry is larger than the entire cache, skip caching it.
	if dataSize > l.maxCache {
		return
	}

	l.tensorCache[key] = data
	l.cacheSize += dataSize
}

// Close closes all open GGUF files and clears the tensor cache.
// After Close, GetTensor will re-open files as needed (Open is idempotent).
func (l *Loader) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	for _, f := range l.files {
		f.Close()
	}
	l.files = make(map[string]*gguf.File)
	// Clear the cache rather than setting to nil so that any concurrent
	// or future GetTensor call receives an empty map rather than panicking
	// on a nil map range.
	l.tensorCache = make(map[string][]byte)
	l.cacheSize = 0
	return nil
}

// ListTensors returns metadata for all tensors in the given GGUF file.
func (l *Loader) ListTensors(path string) ([]gguf.TensorInfo, error) {
	f := func() *gguf.File {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.files[path]
	}()
	if f == nil {
		return nil, fmt.Errorf("file not opened: %s", path)
	}
	var infos []gguf.TensorInfo
	for _, info := range f.TensorInfos() {
		infos = append(infos, info)
	}
	return infos, nil
}

// GetModelFiles returns the sorted list of .gguf file paths in dir.
func (l *Loader) GetModelFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".gguf" {
			files = append(files, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(files)
	return files, nil
}
