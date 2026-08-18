//go:build linux

package ml

import (
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
	got := GetDevicesEnv(gpus)
	if got["ROCR_VISIBLE_DEVICES"] != "0" {
		t.Fatalf("GetDevicesEnv ROCR for PCI BDF: got %#v want ROCR_VISIBLE_DEVICES=0 (ROCm 7 rejects BDF)", got)
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
	got := GetDevicesEnv(gpus)
	if got["ROCR_VISIBLE_DEVICES"] != "1" {
		t.Fatalf("GetDevicesEnv ROCR numeric: got %#v want 1", got)
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
	got := GetDevicesEnv(gpus)
	if got["ROCR_VISIBLE_DEVICES"] != "1" {
		t.Fatalf("GetDevicesEnv ROCR FilterID: got %#v want 1", got)
	}
}
