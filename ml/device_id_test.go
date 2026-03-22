package ml

import "testing"

func TestDeviceIDMatchesForOffload(t *testing.T) {
	t.Parallel()
	pci := "0000:0b:00.0"
	rocm := DeviceID{ID: pci, Library: "ROCm"}
	vulkan := DeviceID{ID: pci, Library: "Vulkan"}
	if !rocm.MatchesForOffload(vulkan) {
		t.Fatal("same PCI id should match across libraries")
	}
	if !rocm.MatchesForOffload(rocm) {
		t.Fatal("identity should match")
	}
	if rocm.MatchesForOffload(DeviceID{ID: "0000:03:00.0", Library: "Vulkan"}) {
		t.Fatal("different PCI id should not match")
	}
}
