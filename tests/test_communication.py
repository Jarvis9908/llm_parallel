"""通信原语测试。

用 torchrun 启动多进程测试:
    torchrun --nproc_per_node=4 tests/test_communication.py

也可以单独运行某个测试:
    torchrun --nproc_per_node=4 tests/test_communication.py --test all_reduce
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.distributed as dist

from parallel.communication.setup import (
    init_process_group, cleanup, get_rank, get_world_size,
)
from parallel.communication.primitives import (
    naive_all_reduce, ring_all_reduce, naive_all_gather, naive_broadcast,
)


def test_all_reduce_consistency():
    """验证 naive_all_reduce 在所有 rank 上产生一致的求和结果。"""
    rank = get_rank()
    world_size = get_world_size()
    init_process_group(backend="gloo")
    device = torch.device("cpu")

    tensor = torch.tensor([rank * 1.0], device=device)
    result = naive_all_reduce(tensor, op="sum")
    expected = float(sum(range(world_size)))

    assert torch.allclose(result, torch.tensor([expected], device=device)), \
        f"Rank {rank}: expected {expected}, got {result.item()}"

    # 验证 avg 模式
    result_avg = naive_all_reduce(tensor, op="avg")
    expected_avg = sum(range(world_size)) / world_size
    assert torch.allclose(result_avg, torch.tensor([expected_avg], device=device)), \
        f"Rank {rank} avg: expected {expected_avg}, got {result_avg.item()}"

    cleanup()


def test_ring_all_reduce():
    """验证 ring_all_reduce 结果与 naive_all_reduce 一致。"""
    rank = get_rank()
    world_size = get_world_size()
    init_process_group(backend="gloo")
    device = torch.device("cpu")

    # 每个 rank 持有不同值，便于验证归约正确性
    tensor = torch.ones(4, device=device) * (rank + 1)
    result_naive = naive_all_reduce(tensor.clone(), op="sum")
    result_ring = ring_all_reduce(tensor.clone())

    # 浮点累加顺序不同，使用宽松容差
    assert torch.allclose(result_naive, result_ring, rtol=1e-5, atol=1e-6), \
        f"Rank {rank}: ring_all_reduce differs from naive_all_reduce"
    assert result_ring.shape == tensor.shape, \
        f"Rank {rank}: shape mismatch {result_ring.shape} vs {tensor.shape}"

    # 验证含 4 个 rank 时 sum 应为 rank 值之和 (1+2+3+4=10)
    expected_val = world_size * (world_size + 1) / 2.0
    assert torch.allclose(result_ring, torch.full_like(result_ring, expected_val)), \
        f"Rank {rank}: expected {expected_val} for all elements"

    cleanup()


def test_ring_all_reduce_odd_sizes():
    """验证 ring_all_reduce 对非对齐张量大小的处理（需要补零的边界情况）。"""
    rank = get_rank()
    world_size = get_world_size()
    init_process_group(backend="gloo")
    device = torch.device("cpu")

    # 7 个元素，world_size=4 时不能整除，需要内部补零
    tensor = torch.arange(1, 8, device=device, dtype=torch.float32) * (rank + 1)
    result_naive = naive_all_reduce(tensor.clone(), op="sum")
    result_ring = ring_all_reduce(tensor.clone())

    assert torch.allclose(result_naive, result_ring, rtol=1e-5, atol=1e-6), \
        f"Rank {rank}: ring_all_reduce (odd size) differs"
    assert result_ring.shape == tensor.shape, \
        f"Rank {rank}: shape mismatch {result_ring.shape} vs {tensor.shape}"

    cleanup()


def test_broadcast():
    """验证 naive_broadcast 将 src rank 的数据正确分发到所有 rank。"""
    rank = get_rank()
    init_process_group(backend="gloo")
    device = torch.device("cpu")

    if rank == 0:
        tensor = torch.tensor([42.0, 3.14], device=device)
    else:
        tensor = torch.zeros(2, device=device)

    result = naive_broadcast(tensor, src=0)
    assert torch.allclose(result, torch.tensor([42.0, 3.14], device=device)), \
        f"Rank {rank}: broadcast failed, got {result}"

    cleanup()


def test_broadcast_nonzero_src():
    """验证从非 0 号 rank 广播也正确。"""
    rank = get_rank()
    world_size = get_world_size()
    if world_size < 2:
        return  # skip single-process

    init_process_group(backend="gloo")
    device = torch.device("cpu")

    src = world_size - 1  # 最后一个 rank 作为源
    if rank == src:
        tensor = torch.tensor([99.0, 0.5], device=device)
    else:
        tensor = torch.zeros(2, device=device)

    result = naive_broadcast(tensor, src=src)
    assert torch.allclose(result, torch.tensor([99.0, 0.5], device=device)), \
        f"Rank {rank}: broadcast from src={src} failed, got {result}"

    cleanup()


def test_all_gather():
    """验证 naive_all_gather 在所有 rank 上拼接出相同的全局数据。"""
    rank = get_rank()
    world_size = get_world_size()
    init_process_group(backend="gloo")
    device = torch.device("cpu")

    local_data = torch.tensor([rank * 10.0, rank * 10.0 + 1.0], device=device)
    gathered = naive_all_gather(local_data)

    assert gathered.shape[0] == local_data.shape[0] * world_size, \
        f"Rank {rank}: gathered size {gathered.shape[0]}, expected {local_data.shape[0] * world_size}"

    for i in range(world_size):
        assert torch.allclose(gathered[i * 2], torch.tensor(i * 10.0, device=device)), \
            f"Rank {rank}: gathered[{i*2}] expected {i*10.0}"
        assert torch.allclose(gathered[i * 2 + 1], torch.tensor(i * 10.0 + 1.0, device=device)), \
            f"Rank {rank}: gathered[{i*2+1}] expected {i*10.0+1.0}"

    cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all")
    args, _ = parser.parse_known_args()

    tests = {
        "all_reduce": test_all_reduce_consistency,
        "ring_all_reduce": test_ring_all_reduce,
        "ring_all_reduce_odd": test_ring_all_reduce_odd_sizes,
        "broadcast": test_broadcast,
        "broadcast_nonzero_src": test_broadcast_nonzero_src,
        "all_gather": test_all_gather,
    }

    if args.test == "all":
        for name, fn in tests.items():
            fn()
            if get_rank() == 0:
                print(f"  {name} passed")
    else:
        if args.test in tests:
            tests[args.test]()
            if get_rank() == 0:
                print(f"  {args.test} passed")
        else:
            if get_rank() == 0:
                print(f"Unknown test: {args.test}")
                print(f"Available: {list(tests.keys())}")
