//go:build linux

package ml

import (
	"reflect"
	"testing"
)

func TestGetVisibleDevicesEnvROCmPCIUsesID(t *testing.T) {
	t.Parallel()
	gpus := []DeviceInfo{
		{
			DeviceID: DeviceID{ID: "0000:0b:00.0", Library: "ROCm"},
			FilterID: "",
		},
	}
	got := GetVisibleDevicesEnv(gpus, true)
	want := map[string]string{"ROCR_VISIBLE_DEVICES": "0000:0b:00.0"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GetVisibleDevicesEnv: got %#v want %#v", got, want)
	}
}

func TestGetVisibleDevicesEnvROCmNumeric(t *testing.T) {
	t.Parallel()
	gpus := []DeviceInfo{
		{
			DeviceID: DeviceID{ID: "1", Library: "ROCm"},
			FilterID: "",
		},
	}
	got := GetVisibleDevicesEnv(gpus, true)
	want := map[string]string{"ROCR_VISIBLE_DEVICES": "1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GetVisibleDevicesEnv: got %#v want %#v", got, want)
	}
}

func TestGetVisibleDevicesEnvROCmFilterID(t *testing.T) {
	t.Parallel()
	gpus := []DeviceInfo{
		{
			DeviceID: DeviceID{ID: "0", Library: "ROCm"},
			FilterID: "1",
		},
	}
	got := GetVisibleDevicesEnv(gpus, true)
	want := map[string]string{"ROCR_VISIBLE_DEVICES": "1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GetVisibleDevicesEnv: got %#v want %#v", got, want)
	}
}

func TestGetVisibleDevicesEnvCUDAUnfiltered(t *testing.T) {
	t.Parallel()
	gpus := []DeviceInfo{
		{DeviceID: DeviceID{ID: "0", Library: "CUDA"}},
	}
	got := GetVisibleDevicesEnv(gpus, false)
	if len(got) != 0 {
		t.Fatalf("expected empty env when CUDA mustFilter=false, got %#v", got)
	}
}
