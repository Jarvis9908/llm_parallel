"""GPipe 流水线调度。所有 micro-batch 先 forward 完毕，再统一 backward。"""
import torch
from typing import Callable


def gpiped_forward(
    micro_batches: list[torch.Tensor],
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """
    GPipe 风格：所有 micro-batch 依次 forward。
    简单但 activation 显存峰值高（需保存所有 micro-batch 的中间值）。
    """
    return [forward_fn(mb) for mb in micro_batches]


def compute_gpipe_bubble_time(n_micro_batches: int, n_stages: int) -> float:
    """GPipe bubble time: (n_stages - 1) / (n_stages - 1 + n_micro_batches)"""
    return (n_stages - 1) / (n_stages - 1 + n_micro_batches)


if __name__ == "__main__":
    print("=== GPipe demo ===")

    # 模拟 forward
    forward_fn = lambda x: x * 2
    micro_batches = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    outputs = gpiped_forward(micro_batches, forward_fn)
    print(f"GPipe forward outputs: {outputs}")

    # Bubble time 分析
    for n_mb in [4, 8, 16, 32]:
        for n_stages in [2, 4, 8]:
            bubble = compute_gpipe_bubble_time(n_mb, n_stages)
            print(f"n_mb={n_mb:2d}, n_stages={n_stages:2d} -> bubble_time={bubble:.4f}")
