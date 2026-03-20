package distributed

import (
	"context"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestClusterConfig(t *testing.T) {
	config := ClusterConfig{
		Nodes:       []NodeInfo{{ID: "node0"}, {ID: "node1"}},
		Topology:    "ring",
		RDMAEnabled: true,
		GDREnabled:  true,
	}

	if len(config.Nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(config.Nodes))
	}
	if config.Topology != "ring" {
		t.Errorf("expected ring topology, got %s", config.Topology)
	}
	if !config.RDMAEnabled {
		t.Error("expected RDMA enabled")
	}
}

func TestNodeInfo(t *testing.T) {
	node := NodeInfo{
		ID:          "node0",
		Address:     "192.168.1.1:8080",
		GPUs:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0"}}},
		MemoryTotal: 16 * 1024 * 1024 * 1024,
		BandwidthTo: map[string]float64{"node1": 100.0},
	}

	if node.ID != "node0" {
		t.Errorf("expected node0, got %s", node.ID)
	}
	if node.BandwidthTo["node1"] != 100.0 {
		t.Errorf("expected 100.0 bandwidth, got %f", node.BandwidthTo["node1"])
	}
}

func TestShardStrategy(t *testing.T) {
	if LayerShard != 0 {
		t.Errorf("expected LayerShard = 0, got %d", LayerShard)
	}
	if TensorShard != 1 {
		t.Errorf("expected TensorShard = 1, got %d", TensorShard)
	}
	if ExpertShard != 2 {
		t.Errorf("expected ExpertShard = 2, got %d", ExpertShard)
	}
}

func TestCluster_New(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{
			{ID: "node0", Address: "192.168.1.1"},
			{ID: "node1", Address: "192.168.1.2"},
		},
	}

	cluster := NewCluster(config, "node0")
	if cluster == nil {
		t.Fatal("expected non-nil Cluster")
	}
	if len(cluster.nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(cluster.nodes))
	}
	if cluster.localNodeID != "node0" {
		t.Errorf("expected localNodeID node0, got %s", cluster.localNodeID)
	}
}

func TestCluster_Connect(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{
			{ID: "node0", Address: "192.168.1.1"},
			{ID: "node1", Address: "192.168.1.2"},
		},
	}

	cluster := NewCluster(config, "node0")
	err := cluster.Connect(context.Background())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCluster_Disconnect(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{
			{ID: "node0", Address: "192.168.1.1"},
		},
	}

	cluster := NewCluster(config, "node0")
	cluster.Disconnect()
}

func TestCluster_GetNode(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{
			{ID: "node0", Address: "192.168.1.1"},
		},
	}

	cluster := NewCluster(config, "node0")
	node, err := cluster.GetNode("node0")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if node.info.ID != "node0" {
		t.Errorf("expected node0, got %s", node.info.ID)
	}
}

func TestCluster_GetNode_NotFound(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{
			{ID: "node0", Address: "192.168.1.1"},
		},
	}

	cluster := NewCluster(config, "node0")
	_, err := cluster.GetNode("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent node")
	}
}

func TestCluster_SetShardStrategy(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{{ID: "node0"}},
	}

	cluster := NewCluster(config, "node0")
	cluster.SetShardStrategy(TensorShard)

	if cluster.GetShardStrategy() != TensorShard {
		t.Error("expected TensorShard strategy")
	}
}

func TestModelSharder_New(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{{ID: "node0"}, {ID: "node1"}},
	}
	cluster := NewCluster(config, "node0")

	sharder := NewModelSharder(cluster, 4)
	if sharder == nil {
		t.Fatal("expected non-nil ModelSharder")
	}
	if sharder.numShards != 4 {
		t.Errorf("expected 4 shards, got %d", sharder.numShards)
	}
}

func TestModelSharder_GetShardForLayer_LayerShard(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{{ID: "node0"}, {ID: "node1"}},
	}
	cluster := NewCluster(config, "node0")

	sharder := NewModelSharder(cluster, 2)
	sharder.shardStrategy = LayerShard

	nodeID, err := sharder.GetShardForLayer(0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if nodeID != "node0" && nodeID != "node1" {
		t.Errorf("expected valid node ID, got %s", nodeID)
	}
}

func TestDistributedKVCache_New(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{{ID: "node0"}},
	}
	cluster := NewCluster(config, "node0")

	cache := NewDistributedKVCache(cluster, "key")
	if cache == nil {
		t.Fatal("expected non-nil DistributedKVCache")
	}
}

func TestDistributedKVCache_Get_Set(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{{ID: "node0"}},
	}
	cluster := NewCluster(config, "node0")

	cache := NewDistributedKVCache(cluster, "key")
	cache.Set("token_key", []float32{1.0, 2.0, 3.0})

	val, ok := cache.Get("token_key")
	if !ok {
		t.Error("expected to find cached value")
	}
	if len(val) != 3 {
		t.Errorf("expected 3 values, got %d", len(val))
	}
}

func TestDistributedKVCache_Get_NotFound(t *testing.T) {
	config := ClusterConfig{
		Nodes: []NodeInfo{{ID: "node0"}},
	}
	cluster := NewCluster(config, "node0")

	cache := NewDistributedKVCache(cluster, "key")
	_, ok := cache.Get("nonexistent")
	if ok {
		t.Error("expected not to find nonexistent key")
	}
}

func TestRDMAClient_New(t *testing.T) {
	client := NewRDMAClient(0)
	if client == nil {
		t.Fatal("expected non-nil RDMAClient")
	}
	if client.deviceID != 0 {
		t.Errorf("expected deviceID 0, got %d", client.deviceID)
	}
}

func TestRDMAClient_Connect(t *testing.T) {
	client := NewRDMAClient(0)
	err := client.Connect("192.168.1.1:12345")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !client.IsConnected() {
		t.Error("expected to be connected")
	}
}

func TestNCCLComm_New(t *testing.T) {
	comm := NewNCCLComm("comm_id", 4, 0)
	if comm == nil {
		t.Fatal("expected non-nil NCCLComm")
	}
	if comm.numProcs != 4 {
		t.Errorf("expected 4 procs, got %d", comm.numProcs)
	}
	if comm.procid != 0 {
		t.Errorf("expected procid 0, got %d", comm.procid)
	}
}

func TestNCCLComm_Init(t *testing.T) {
	comm := NewNCCLComm("comm_id", 4, 0)
	err := comm.Init()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !comm.IsInitialized() {
		t.Error("expected to be initialized")
	}
}

func TestNCCLComm_Barrier(t *testing.T) {
	comm := NewNCCLComm("comm_id", 4, 0)
	comm.Init()
	err := comm.Barrier()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestNCCLComm_Finalize(t *testing.T) {
	comm := NewNCCLComm("comm_id", 4, 0)
	comm.Init()
	err := comm.Finalize()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if comm.IsInitialized() {
		t.Error("expected not to be initialized after finalize")
	}
}
