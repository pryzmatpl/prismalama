package qwen3next

import (
	"iter"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs"
)

func TestMropePositionIDs(t *testing.T) {
	got := mropePositionIDs([]int32{0, 1, 2})
	want := []int32{
		0, 1, 2, // time
		0, 1, 2, // height
		0, 1, 2, // width
		0, 0, 0, // extra
	}
	if !slices.Equal(got, want) {
		t.Fatalf("mropePositionIDs = %v, want %v", got, want)
	}
	if len(mropePositionIDs(nil)) != 0 {
		t.Fatal("empty positions must stay empty")
	}
}

func TestIntSectionsPadsFourthAxis(t *testing.T) {
	got := intSections([]int32{11, 11, 10})
	want := []int{11, 11, 10, 0}
	if !slices.Equal(got, want) {
		t.Fatalf("intSections = %v, want %v", got, want)
	}
	if intSections(nil) != nil {
		t.Fatal("empty sections must stay nil")
	}
}

type stubConfig struct {
	ints map[string][]int32
}

func (stubConfig) Architecture() string                 { return "qwen35moe" }
func (stubConfig) String(string, ...string) string      { return "" }
func (stubConfig) Uint(string, ...uint32) uint32        { return 0 }
func (stubConfig) Float(string, ...float32) float32     { return 0 }
func (stubConfig) Bool(string, ...bool) bool            { return false }
func (stubConfig) Strings(string, ...[]string) []string { return nil }
func (c stubConfig) Ints(key string, _ ...[]int32) []int32 {
	return c.ints[key]
}
func (stubConfig) Floats(string, ...[]float32) []float32 { return nil }
func (stubConfig) Bools(string, ...[]bool) []bool        { return nil }
func (stubConfig) Len() int                              { return 0 }
func (stubConfig) Keys() iter.Seq[string]                { return func(func(string) bool) {} }
func (stubConfig) Value(string) any                      { return nil }

func TestRopeSectionsFromConfig(t *testing.T) {
	var _ fs.Config = stubConfig{}

	got := ropeSectionsFromConfig(stubConfig{ints: map[string][]int32{
		"rope.dimension_sections": {11, 11, 10, 0},
	}})
	if !slices.Equal(got, []int{11, 11, 10, 0}) {
		t.Fatalf("dimension_sections = %v", got)
	}

	got = ropeSectionsFromConfig(stubConfig{ints: map[string][]int32{
		"mrope_sections": {11, 11, 10},
	}})
	if !slices.Equal(got, []int{11, 11, 10, 0}) {
		t.Fatalf("mrope_sections pad = %v", got)
	}

	if ropeSectionsFromConfig(stubConfig{}) != nil {
		t.Fatal("missing sections must disable IMRoPE")
	}
}

func TestUsesIMRoPE(t *testing.T) {
	if (Options{}).usesIMRoPE() {
		t.Fatal("qwen3next without sections stays NeoX")
	}
	if !(Options{mropeSections: []int{11, 11, 10, 0}}).usesIMRoPE() {
		t.Fatal("qwen35moe sections must enable IMRoPE")
	}
}
