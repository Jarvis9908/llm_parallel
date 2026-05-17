"""通信测试启动器。

在 Windows 环境下 PyTorch 2.12 的 TCPStore 默认 use_libuv=True，
但该版本未编译 libuv 支持。本启动器使用 multiprocessing.spawn
替代 torchrun，通过文件存储（FileStore）进行 rendezvous，规避 TCPStore libuv 问题。

用法:
    python tests/run_communication_tests.py                    # 运行全部测试 (4 进程)
    python tests/run_communication_tests.py --test all_reduce  # 运行单个测试
    python tests/run_communication_tests.py --nproc 2          # 指定进程数

在 Linux / macOS 等标准环境下可直接使用 torchrun:
    torchrun --nproc_per_node=4 tests/test_communication.py
"""
import os
import sys
import tempfile

# ── 必须在任何 torch 导入之前设置 ──
os.environ.setdefault("USE_LIBUV", "0")

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# 共享文件路径，用于 FileStore rendezvous
FILE_STORE_PATH = os.path.join(tempfile.gettempdir(), "llm_parallel_comm_test_store")


def _worker(rank, world_size, test_name, project_root):
    """每个进程的入口：设置环境变量 -> 初始化进程组 -> 运行测试 -> 清理。"""
    # 确保子进程能导入 parallel 模块
    sys.path.insert(0, project_root)

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # 使用 FileStore 进行 rendezvous，避免 Windows 上 TCPStore 的 libuv 兼容性问题
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{FILE_STORE_PATH}",
        rank=rank,
        world_size=world_size,
    )

    from parallel.communication.primitives import (
        naive_all_reduce, ring_all_reduce, naive_all_gather, naive_broadcast,
    )

    if test_name == "all" or test_name == "all_reduce":
        tensor = torch.tensor([rank * 1.0])
        result = naive_all_reduce(tensor, op="sum")
        expected = float(sum(range(world_size)))
        assert torch.allclose(result, torch.tensor([expected])), \
            f"Rank {rank}: all_reduce expected {expected}, got {result.item()}"
        if rank == 0:
            print("  all_reduce passed")

    if test_name == "all" or test_name == "ring_all_reduce":
        tensor = torch.ones(4) * (rank + 1)
        result_naive = naive_all_reduce(tensor.clone(), op="sum")
        result_ring = ring_all_reduce(tensor.clone())
        assert torch.allclose(result_naive, result_ring, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}: ring_all_reduce differs from naive_all_reduce"
        expected_val = world_size * (world_size + 1) / 2.0
        assert torch.allclose(result_ring, torch.full_like(result_ring, expected_val)), \
            f"Rank {rank}: ring_all_reduce expected {expected_val}"
        if rank == 0:
            print("  ring_all_reduce passed")

    if test_name == "all" or test_name == "ring_all_reduce_odd":
        tensor = torch.arange(1, 8, dtype=torch.float32) * (rank + 1)
        result_naive = naive_all_reduce(tensor.clone(), op="sum")
        result_ring = ring_all_reduce(tensor.clone())
        assert torch.allclose(result_naive, result_ring, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}: ring_all_reduce (odd size) differs from naive"
        if rank == 0:
            print("  ring_all_reduce_odd passed")

    if test_name == "all" or test_name == "broadcast":
        if rank == 0:
            tensor = torch.tensor([42.0, 3.14])
        else:
            tensor = torch.zeros(2)
        result = naive_broadcast(tensor, src=0)
        assert torch.allclose(result, torch.tensor([42.0, 3.14])), \
            f"Rank {rank}: broadcast failed"
        if rank == 0:
            print("  broadcast passed")

    if test_name == "all" or test_name == "broadcast_nonzero_src":
        if world_size >= 2:
            src = world_size - 1
            if rank == src:
                tensor = torch.tensor([99.0, 0.5])
            else:
                tensor = torch.zeros(2)
            result = naive_broadcast(tensor, src=src)
            assert torch.allclose(result, torch.tensor([99.0, 0.5])), \
                f"Rank {rank}: broadcast src={src} failed"
            if rank == 0:
                print("  broadcast_nonzero_src passed")

    if test_name == "all" or test_name == "all_gather":
        local_data = torch.tensor([rank * 10.0, rank * 10.0 + 1.0])
        gathered = naive_all_gather(local_data)
        assert gathered.shape[0] == local_data.shape[0] * world_size, \
            f"Rank {rank}: all_gather size mismatch"
        for i in range(world_size):
            assert torch.allclose(gathered[i * 2], torch.tensor(i * 10.0)), \
                f"Rank {rank}: all_gather[{i*2}] wrong"
            assert torch.allclose(gathered[i * 2 + 1], torch.tensor(i * 10.0 + 1.0)), \
                f"Rank {rank}: all_gather[{i*2+1}] wrong"
        if rank == 0:
            print("  all_gather passed")

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all")
    parser.add_argument("--nproc", type=int, default=4)
    args = parser.parse_args()

    world_size = args.nproc
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 清理上一次测试残留的文件存储
    if os.path.exists(FILE_STORE_PATH):
        os.remove(FILE_STORE_PATH)

    print(f"Running communication tests with {world_size} processes (test={args.test})")
    mp.spawn(_worker, args=(world_size, args.test, project_root), nprocs=world_size, join=True)
    print("All communication tests passed.")

    # 清理文件存储
    if os.path.exists(FILE_STORE_PATH):
        os.remove(FILE_STORE_PATH)
