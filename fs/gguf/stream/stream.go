package stream

import (
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/ollama/ollama/fs/gguf"
)

type Loader struct {
	files       map[string]*gguf.File
	tensorCache map[string][]byte
	mu          sync.RWMutex
	maxCache    int64
	cacheSize   int64
}

func NewLoader(maxCacheBytes int64) *Loader {
	return &Loader{
		files:       make(map[string]*gguf.File),
		tensorCache: make(map[string][]byte),
		maxCache:    maxCacheBytes,
	}
}

func (l *Loader) Open(path string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

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

func (l *Loader) GetTensor(filePath, tensorName string) ([]byte, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	cacheKey := filePath + ":" + tensorName

	if data, ok := l.tensorCache[cacheKey]; ok {
		return data, nil
	}

	f, ok := l.files[filePath]
	if !ok {
		return nil, fmt.Errorf("file not opened: %s", filePath)
	}

	info, reader, err := f.TensorReader(tensorName)
	if err != nil {
		return nil, err
	}

	data := make([]byte, info.NumBytes())
	n, err := io.ReadFull(reader, data)
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		return nil, fmt.Errorf("failed to read tensor %s: %w", tensorName, err)
	}
	data = data[:n]

	l.addToCache(cacheKey, data)

	return data, nil
}

func (l *Loader) addToCache(key string, data []byte) {
	for l.cacheSize+l.maxCache > l.maxCache && len(l.tensorCache) > 0 {
		var oldestKey string
		var oldestSize int64
		for k, v := range l.tensorCache {
			oldestKey = k
			oldestSize = int64(len(v))
			break
		}
		if oldestKey == "" {
			break
		}
		delete(l.tensorCache, oldestKey)
		l.cacheSize -= oldestSize
	}

	l.tensorCache[key] = data
	l.cacheSize += int64(len(data))
}

func (l *Loader) ListTensors(filePath string) (map[string]gguf.TensorInfo, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	f, ok := l.files[filePath]
	if !ok {
		return nil, fmt.Errorf("file not opened: %s", filePath)
	}

	result := make(map[string]gguf.TensorInfo)
	for _, t := range f.TensorInfos() {
		result[t.Name] = t
	}
	return result, nil
}

func (l *Loader) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	for _, f := range l.files {
		f.Close()
	}
	l.files = nil
	l.tensorCache = nil
	return nil
}

func (l *Loader) GetModelFiles(modelPath string) ([]string, error) {
	entries, err := os.ReadDir(modelPath)
	if err != nil {
		return nil, err
	}

	var ggufFiles []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if len(name) > 5 && name[len(name)-5:] == ".gguf" {
			ggufFiles = append(ggufFiles, name)
		}
	}
	return ggufFiles, nil
}
