"""流水线并行的层切分。将模型的 TransformerBlock 按层分配到不同 rank。"""
import torch


def partition_layers(layers: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """简化的层分配：所有层都在本地（单机演示用）"""
    return layers


def get_layer_range(n_layers: int, rank: int, world_size: int) -> tuple[int, int]:
    """计算当前 rank 负责的层范围 [start, end)"""
    layers_per_rank = n_layers // world_size
    remainder = n_layers % world_size
    start = rank * layers_per_rank + min(rank, remainder)
    end = start + layers_per_rank + (1 if rank < remainder else 0)
    return start, end


if __name__ == "__main__":
    print("=== layer_partition demo ===")
    # 模拟 12 层 Transformer 分配到 4 个 rank
    n_layers = 12
    world_size = 4
    for rank in range(world_size):
        start, end = get_layer_range(n_layers, rank, world_size)
        print(f"Rank {rank}: layers [{start}:{end}) ({end - start} layers)")

    # 模拟 13 层非均匀分配
    n_layers = 13
    print(f"\nUneven partition ({n_layers} layers, {world_size} ranks):")
    for rank in range(world_size):
        start, end = get_layer_range(n_layers, rank, world_size)
        print(f"Rank {rank}: layers [{start}:{end}) ({end - start} layers)")
