"""
梯度累积 (Gradient Accumulation)。

在不增加显存的情况下模拟更大的 batch size。
例如：目标 batch_size=256，实际每步只能跑 64，
     累积 4 步后同步一次梯度，等效于 batch_size=256。

这在 DP 场景中非常重要：它减少了通信频率（每 accumulation_steps 步才同步一次），
从而降低通信开销占总计算时间的比例。
"""
import torch


class GradientAccumulator:
    """梯度累积器。累积 micro-batch 的梯度，达到 steps 后触发同步。

    使用方式：
        acc = GradientAccumulator(model, accumulation_steps=4)
        for micro_batch in dataloader:
            loss = compute_loss(model, micro_batch)
            loss.backward()  # 梯度累积到 param.grad 上
            if acc.step():    # 达到 accumulation_steps 时返回 True
                optimizer.step()
                optimizer.zero_grad()
    """

    def __init__(self, model: torch.nn.Module, accumulation_steps: int):
        """
        Args:
            model: 被训练的模型。
            accumulation_steps: 累积多少步后执行一次参数更新。
                               必须 > 0。
        """
        if accumulation_steps <= 0:
            raise ValueError(
                f"accumulation_steps 必须 > 0，当前值: {accumulation_steps}"
            )
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def step(self) -> bool:
        """
        执行一步。返回 True 表示应该做 optimizer.step() + zero_grad()。

        流程：
        1. loss.backward()（梯度累积到 param.grad）
        2. 调用 step()
        3. 当累积步数达到 accumulation_steps 时返回 True

        注意：梯度会按 accumulation_steps 取平均，等效于一次大 batch 的 forward。
              这样 loss 的数值量级与不使用累积时保持一致，
              便于学习率等超参数的调优。

        Returns:
            True 表示本次累积已完成，应执行 optimizer.step() 和 zero_grad()。
            False 表示还需继续累积。
        """
        self.current_step += 1
        if self.current_step % self.accumulation_steps == 0:
            # 对累积的梯度取平均
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data /= self.accumulation_steps
            return True
        return False

    def reset(self):
        """重置累积计数。通常在 epoch 开始时调用。"""
        self.current_step = 0


def compute_effective_batch_size(
    micro_batch_size: int, accumulation_steps: int, world_size: int
) -> int:
    """
    计算 DP + 梯度累积下的等效全局 batch size。

    等效 batch size = micro_batch_size * accumulation_steps * world_size

    例如：micro=64, accum=4, gpus=8 -> 等效 batch size = 2048

    理解等效 batch size 对训练至关重要：
    - 学习率通常需要随等效 batch size 线性缩放（linear scaling rule）。
    - 过大的等效 batch size 可能导致泛化性能下降。
    - 过小的等效 batch size 可能使训练不稳定。

    Args:
        micro_batch_size: 每个 GPU 每次 forward 处理的样本数。
        accumulation_steps: 梯度累积步数。
        world_size: GPU 数量（数据并行维度）。

    Returns:
        等效全局 batch size。
    """
    return micro_batch_size * accumulation_steps * world_size


if __name__ == "__main__":
    print("=" * 60)
    print("梯度累积演示")
    print("=" * 60)

    # 构建简单模型
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    accumulation_steps = 4
    micro_batch_size = 8
    world_size = 2  # 假设 2 卡 DP

    acc = GradientAccumulator(model, accumulation_steps)

    print(f"\n配置:")
    print(f"  micro_batch_size = {micro_batch_size}")
    print(f"  accumulation_steps = {accumulation_steps}")
    print(f"  world_size (DP) = {world_size}")
    effective = compute_effective_batch_size(
        micro_batch_size, accumulation_steps, world_size
    )
    print(f"  等效全局 batch size = {effective}")

    print(f"\n模拟训练循环 (共 {accumulation_steps} 步):")
    print("-" * 40)

    for i in range(accumulation_steps):
        # 模拟 micro-batch 训练
        x = torch.randn(micro_batch_size, 4)
        target = torch.randn(micro_batch_size, 2)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()

        should_update = acc.step()
        print(
            f"  Step {i + 1}/{accumulation_steps}: "
            f"loss={loss.item():.4f}, "
            f"should_update={should_update}"
        )

        if should_update:
            # 此时梯度已除以 accumulation_steps
            grad_mean = model.weight.grad.mean().item()
            print(f"  执行 optimizer.step()! grad_mean={grad_mean:.8f}")
            optimizer.step()
            optimizer.zero_grad()

    print("\n梯度累积的核心价值:")
    print("  1. 突破显存限制，在小 GPU 上训练大 batch")
    print("  2. 在 DP 中减少通信频率（accumulation_steps 次 forward 才一次同步）")
    print("  3. 稳定训练：等效大 batch 的梯度估计方差更小")
