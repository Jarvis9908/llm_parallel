"""并行策略可视化工具：绘制通信拓扑图和流水线气泡时间对比图。"""
import os

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，避免在没有 GUI 的环境中报错
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _require_matplotlib():
    """检查 matplotlib 是否可用，不可用时抛出明确错误提示。"""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib 未安装，无法使用可视化功能。"
            "请运行: pip install matplotlib"
        )


def plot_topology(nodes: int, topology: str = "ring", save_path: str = None):
    """
    绘制 GPU 通信拓扑图。

    用圆形布局绘制指定数量节点的拓扑结构。当前支持 ring 拓扑，
    每个节点表示一个 GPU，连线表示通信链路。
    需要安装 matplotlib 库。

    Args:
        nodes: 节点数量（GPU 数量）
        topology: 拓扑类型，当前支持 "ring"（环形拓扑）
        save_path: 保存路径，为 None 则仅在终端输出提示信息
    """
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))
    if topology == "ring":
        import numpy as np
        angles = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        ax.scatter(x, y, s=300, c='lightblue', edgecolors='navy')
        for i in range(nodes):
            j = (i + 1) % nodes
            ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=2)
        for i in range(nodes):
            ax.annotate(str(i), (x[i], y[i]), ha='center', va='center', fontsize=12)
    ax.set_title(f"{topology.title()} 拓扑结构 ({nodes} 个节点)", fontsize=14)
    ax.set_aspect('equal')
    ax.axis('off')
    if save_path:
        plt.savefig(save_path)
        print(f"拓扑图已保存至 {save_path}")
    else:
        print("(拓扑图已生成 — 传入 save_path 参数即可保存为文件)")


def plot_bubble_time(n_micro_batches_range: list, n_stages: int):
    """
    绘制 GPipe 与 1F1B 调度策略的气泡时间对比图。

    GPipe 是所有 micro-batch 前向完成后再反向，空闲时间较多；
    1F1B（one-forward-one-backward）交替执行前向和反向，减少空闲等待。
    气泡时间比例越低，GPU 利用率越高。
    需要安装 matplotlib 库。

    GPipe 气泡比例: (n_stages - 1) / (n_stages - 1 + n_micro_batches)
    1F1B 气泡比例: 2*(n_stages - 1) / (2*(n_stages - 1 + n_micro_batches) - 1)

    Args:
        n_micro_batches_range: micro-batch 数量列表，如 [1, 2, 4, 8, 16, 32]
        n_stages: 流水线阶段数（GPU 或层分组数）

    Returns:
        matplotlib.figure.Figure or None: 气泡时间对比图，matplotlib 不可用时返回 None
    """
    _require_matplotlib()
    # GPipe 气泡时间比例
    gp_bubbles = [(n_stages - 1) / (n_stages - 1 + m) for m in n_micro_batches_range]
    # 1F1B 气泡时间比例
    f1b1_bubbles = [2 * (n_stages - 1) / (2 * (n_stages - 1 + m) - 1)
                    for m in n_micro_batches_range]

    fig, ax = plt.subplots()
    ax.plot(n_micro_batches_range, gp_bubbles, 'o-', label='GPipe')
    ax.plot(n_micro_batches_range, f1b1_bubbles, 's-', label='1F1B')
    ax.set_xlabel('Micro-batch 数量')
    ax.set_ylabel('气泡时间比例')
    ax.set_title(f'流水线气泡时间对比 (流水线阶段数={n_stages})')
    ax.legend()
    ax.grid(True)
    return fig


if __name__ == "__main__":
    print("=== 可视化工具演示 ===\n")

    # 1. 绘制 8 节点的 Ring 拓扑图
    if _HAS_MATPLOTLIB:
        plot_topology(8, "ring")
    else:
        print("(跳过拓扑图绘制 — matplotlib 未安装)")

    # 2. GPipe vs 1F1B 气泡时间对比（控制台输出）
    n_stages = 4
    n_micro_batches_list = [1, 2, 4, 8, 16, 32]
    print(f"\nGPipe vs 1F1B 气泡对比（阶段数={n_stages}）：")
    for m in n_micro_batches_list:
        gp_bubble = (n_stages - 1) / (n_stages - 1 + m)
        f1b1_bubble = 2 * (n_stages - 1) / (2 * (n_stages - 1 + m) - 1)
        print(f"  Micro-batch={m:2d}: GPipe={gp_bubble:.3f}, 1F1B={f1b1_bubble:.3f}")

    if _HAS_MATPLOTLIB:
        fig = plot_bubble_time(n_micro_batches_list, n_stages=n_stages)
        print("\n(气泡时间对比图已生成)")

    print("\n可视化工具演示完成")
