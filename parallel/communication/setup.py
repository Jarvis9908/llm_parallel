"""分布式通信环境搭建。单机多进程模拟多 GPU 拓扑。"""

import os

import torch
import torch.distributed as dist


def init_process_group(backend: str = "gloo"):
    """初始化分布式进程组。

    在分布式训练开始时调用，建立所有进程之间的通信通道。
    使用 gloo 后端支持 CPU 通信，无需 GPU 即可在单机上模拟多卡拓扑。
    重复调用是安全的，已初始化时不会重复创建。

    Args:
        backend: 通信后端，可选 "gloo"（CPU）、"nccl"（GPU）。
                 单机模拟使用 "gloo"。
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)


def get_rank() -> int:
    """获取当前进程在分布式拓扑中的序号（rank）。

    在数据并行中用于确定每个进程处理哪部分数据；
    在模型并行中用于确定每个进程持有哪部分模型参数。
    未初始化时回退到环境变量 LOCAL_RANK，默认值为 0，
    以便单进程调试时无需启动 torchrun。

    Returns:
        当前进程的 rank，从 0 开始编号。
    """
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """获取分布式拓扑中的进程总数（world size）。

    在 ring all-reduce 等集合通信算法中用于计算数据分片策略；
    在数据并行中用于梯度同步时取平均。
    未初始化时回退到环境变量 WORLD_SIZE，默认值为 1。

    Returns:
        参与分布式训练的进程总数。
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def cleanup():
    """销毁分布式进程组，释放通信资源。

    在训练结束或测试用例末尾调用，确保所有进程正确退出。
    多次调用是安全的，会先检查是否已初始化。
    """
    if dist.is_initialized():
        dist.destroy_process_group()
