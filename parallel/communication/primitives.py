"""手写分布式通信原语实现。

基于 torch.distributed 底层 API（broadcast、isend、irecv）手写实现
常见的集合通信操作，用于理解分布式训练中数据同步的工作原理。
每种原语都有对应的生产级实现（如 NCCL 的 all_reduce），
本模块关注教学目的——揭示通信模式、复杂度与正确性约束。
"""

import torch
import torch.distributed as dist

from .setup import get_rank, get_world_size


def naive_all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """naive all-reduce：逐 rank 广播并累加。

    原理：
        依次让每个 rank 将其本地张量广播到所有其他 rank，
        每个 rank 将收到的所有张量按 op 规约，得到全局结果。
        通信复杂度 O(N * P^2)，仅用于教学对比，生产环境使用
        NCCL 的 ring all-reduce 或 tree all-reduce。

    在分布式训练中的用途：
        数据并行中，每个 worker 计算出本地梯度后，需要用 all-reduce
        将所有 worker 的梯度求和/平均，然后更新模型参数。
        这是数据并行中唯一的通信瓶颈。

    Args:
        tensor: 当前 rank 的本地张量。
        op: 规约操作，支持 "sum"（求和，默认）和 "avg"（平均）。

    Returns:
        全局规约后的张量，在所有 rank 上结果一致。
    """
    rank = get_rank()
    world_size = get_world_size()

    result = torch.zeros_like(tensor)
    for src in range(world_size):
        buf = tensor.clone() if src == rank else torch.zeros_like(tensor)
        dist.broadcast(buf, src=src)
        if op == "sum":
            result += buf
        elif op == "avg":
            result += buf / world_size
        else:
            raise ValueError(f"Unsupported op: {op}. Use 'sum' or 'avg'.")

    return result


def ring_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """ring all-reduce：带宽最优的环形两阶段算法。

    原理：
        将张量均分为 P 份（P = world_size），通过环形拓扑分两阶段完成：

        阶段一（Reduce-Scatter）：共 P-1 步。
          每步每个 rank 向右侧邻居发送一个分片，同时从左侧邻居接收一个分片并累加。
          经过 P-1 步后，每个 rank 持有一个完整规约后的分片。

        阶段二（All-Gather）：共 P-1 步。
          每步每个 rank 向右侧邻居发送一个已规约分片，同时从左侧邻居接收一个分片并覆盖。
          经过 P-1 步后，每个 rank 拥有所有规约后的分片。

        总通信量 2N，为带宽最优方案（与进程数无关）。
        使用异步发送/接收（isend/irecv）避免死锁。

    在分布式训练中的用途：
        这是 Horovod、NCCL 等框架中 all-reduce 的核心算法。
        在数据并行的梯度同步中，ring all-reduce 保证了通信时间
        不随 GPU 数量增长而线性增加，是大规模分布式训练的关键技术。

    注意事项：
        - 张量长度不足 world_size 时会自动补零对齐，结果裁剪回原始形状。
        - 因浮点累加顺序不同，本函数结果与 naive_all_reduce 可能有微小数值差异。

    Args:
        tensor: 当前 rank 的本地张量。

    Returns:
        全局求和后的张量，在所有 rank 上结果一致。
    """
    rank = get_rank()
    world_size = get_world_size()

    if world_size == 1:
        return tensor.clone()

    shape = tensor.shape
    N = tensor.numel()

    # 确保张量长度可被 world_size 整除，不足则补零
    chunk_size = (N + world_size - 1) // world_size
    padded_size = chunk_size * world_size
    if N < padded_size:
        padded = torch.cat([tensor.flatten(), torch.zeros(padded_size - N)])
    else:
        padded = tensor.flatten().clone()

    chunks = list(padded.split(chunk_size))

    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    # --- 阶段一：Reduce-Scatter ---
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        recv_buf = torch.zeros(chunk_size)

        send_req = dist.isend(chunks[send_idx], dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        send_req.wait()
        recv_req.wait()

        chunks[recv_idx] += recv_buf

    # --- 阶段二：All-Gather ---
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        recv_buf = torch.zeros(chunk_size)

        send_req = dist.isend(chunks[send_idx], dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        send_req.wait()
        recv_req.wait()

        chunks[recv_idx] = recv_buf

    result = torch.cat(chunks)[:N]
    return result.reshape(shape)


def naive_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """naive all-gather：逐 rank 广播并拼接所有 rank 的数据。

    原理：
        依次让每个 rank 广播其本地张量，所有 rank 收集后沿第 0 维拼接。
        最终每个 rank 拥有的张量长度 = 本地长度 * world_size，
        内容按 rank 顺序排列。

    在分布式训练中的用途：
        - 数据并行中，当需要在所有 worker 之间共享预测结果时使用。
        - 模型并行中，当需要收集各分片的输出以计算最终结果时使用。
        - 分布式评估时收集所有 rank 的指标进行汇总。

    Args:
        tensor: 当前 rank 的本地张量（1D）。

    Returns:
        拼接后的全局张量，长度为 len(tensor) * world_size。
    """
    rank = get_rank()
    world_size = get_world_size()

    gathered = []
    for src in range(world_size):
        buf = tensor.clone() if src == rank else torch.zeros_like(tensor)
        dist.broadcast(buf, src=src)
        gathered.append(buf)

    return torch.cat(gathered, dim=0)


def naive_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """broadcast：将源 rank 的数据分发到所有其他 rank。

    原理：
        源 rank（src）将它的张量发送到所有其他 rank，
        非源 rank 接收后获得与源 rank 完全相同的副本。
        直接封装 torch.distributed.broadcast。

    在分布式训练中的用途：
        - 数据并行开始时，将模型参数从 rank 0 广播到所有 worker，
          确保所有进程从相同的初始权重开始训练。
        - 在每个 epoch 开始时广播随机种子，确保数据增强一致。
        - 广播全局配置或超参数到所有节点。

    Args:
        tensor: 源 rank 上的数据张量，非源 rank 上可为任意占位张量（shape 需一致）。
        src: 源 rank 编号，默认为 0。

    Returns:
        广播后的张量，在所有 rank 上值与 src rank 一致。
    """
    data = tensor.clone()
    dist.broadcast(data, src=src)
    return data
