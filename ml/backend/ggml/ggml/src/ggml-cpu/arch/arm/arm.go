//go:build arm64

package arm

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../../../../../../../../llama/llama.cpp/ggml/src/ggml-cpu -I${SRCDIR}/../../../../../../../../llama/llama.cpp/ggml/src -I${SRCDIR}/../../../../../../../../llama/llama.cpp/ggml/include -DHWCAP2_SVE2="2"
import "C"
