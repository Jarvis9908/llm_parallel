"""分布式通信拓扑分析与可视化。

提供三种常见分布式通信拓扑的分析函数——环形（Ring）、树形（Tree）、
二维网格（Mesh），以及对应的 ASCII 图示工具。每个函数返回包含通信步数、
数据量和中文描述的字典，便于教学和性能对比。
"""

import math


def analyze_ring_topology(world_size: int) -> dict:
    """分析环形拓扑的 all-reduce 通信代价。

    原理：
        环形拓扑是 all-reduce 最常用的实现方式，分为两个阶段：

        阶段一（Reduce-Scatter）：共 P-1 步。每步每个节点向右邻居发送
        一个数据分片并接收左邻居的分片进行累加。P-1 步后每个节点持有
        一个完整的规约后分片（总数据的 1/P）。

        阶段二（All-Gather）：共 P-1 步。每步每个节点将已规约的分片发送
        给右邻居，同时接收左邻居的新分片。P-1 步后所有节点拥有全部规约结果。

        总通信量为 2N（N 为总数据量），与进程数 P 无关，是带宽最优方案。
        NCCL 和 Horovod 都基于此算法实现 all-reduce。

    在分布式训练中的用途：
        数据并行中每个 worker 计算本地梯度后，需要通过 ring all-reduce
        同步梯度。环形拓扑的带宽最优特性保证通信时间不随 GPU 数量增加而线性增长。

    Args:
        world_size: 分布式环境中的进程总数 P。

    Returns:
        包含以下字段的字典：
        - steps_reduce_scatter: Reduce-Scatter 阶段步数 (P-1)
        - steps_all_gather: All-Gather 阶段步数 (P-1)
        - total_steps: 总通信步数 2(P-1)
        - per_step_data: 每步传输的数据量占总数据的比例 (1/P)
        - description: 中文描述字符串
    """
    if world_size < 1:
        raise ValueError(f"world_size 必须 >= 1，当前值: {world_size}")

    steps = world_size - 1
    return {
        "steps_reduce_scatter": steps,
        "steps_all_gather": steps,
        "total_steps": 2 * steps,
        "per_step_data": f"1/{world_size} (即总数据的 {1/world_size:.2%})" if world_size > 1 else "N/A",
        "description": (
            f"环形拓扑 all-reduce：{world_size} 个进程通过环形连接，"
            f"Reduce-Scatter 需 {steps} 步，All-Gather 需 {steps} 步，"
            f"共 {2 * steps} 步。每步传输总数据的 1/{world_size}，"
            f"总通信量恒为 2N，与进程数无关（带宽最优）。"
        ),
    }


def analyze_tree_topology(world_size: int) -> dict:
    """分析二叉树拓扑的 broadcast 通信代价。

    原理：
        二叉树拓扑以 rank 0 为根节点构建一棵近似完全二叉树。
        根节点将数据发送给两个子节点，子节点再转发给各自的子节点，
        逐层扩散，直到底层的叶子节点。

        树高 = ceil(log2(P))，即从根到最远叶子节点所需的跳数。
        这也是 broadcast 所需的总步数（同一层内的节点可并行发送）。

        每个内部节点最多有 2 个子节点，叶子节点不转发数据。
        总共有 ceil(P/2) - 1 个内部转发节点和 ceil(P/2) 个叶子节点。

    在分布式训练中的用途：
        - 模型参数初始化广播：rank 0 加载预训练权重后，通过树形拓扑
          高效分发到所有 worker，避免根节点成为串行瓶颈。
        - 超参数同步：将学习率、批次大小等配置从主节点分发到所有 worker。

    Args:
        world_size: 分布式环境中的进程总数 P。

    Returns:
        包含以下字段的字典：
        - tree_height: 二叉树高度 ceil(log2(P))
        - total_steps_for_broadcast: broadcast 所需步数（等于树高）
        - max_children_per_node: 每个节点最大子节点数 (2)
        - description: 中文描述字符串
    """
    if world_size < 1:
        raise ValueError(f"world_size 必须 >= 1，当前值: {world_size}")

    if world_size == 1:
        tree_height = 0
    else:
        tree_height = math.ceil(math.log2(world_size))

    internal_nodes = (world_size + 1) // 2 - 1 if world_size > 1 else 0
    leaf_nodes = world_size - internal_nodes

    return {
        "tree_height": tree_height,
        "total_steps_for_broadcast": tree_height,
        "max_children_per_node": 2,
        "internal_nodes": internal_nodes,
        "leaf_nodes": leaf_nodes,
        "description": (
            f"二叉树拓扑 broadcast：{world_size} 个进程按二叉树组织，"
            f"树高 {tree_height} 层，共需 {tree_height} 步完成广播。"
            f"每个内部节点最多 2 个子节点，{internal_nodes} 个内部节点负责转发，"
            f"{leaf_nodes} 个叶子节点只接收不转发。"
            f"时间复杂度 O(log P)，远优于朴素广播的 O(P)。"
        ),
    }


def analyze_mesh_topology(tp_size: int, dp_size: int) -> dict:
    """分析 Megatron 风格的二维网格 (TP+DP) 通信代价。

    原理：
        二维网格将 tp_size × dp_size 个设备排列成 tp_size 行 × dp_size 列的网格。
        这是 Megatron-LM 中张量并行（TP）与数据并行（DP）混合策略的标准拓扑：

        - 每行的 tp_size 个设备组成一个张量并行组（TP group），共同处理
          同一个样本的不同模型分片。TP 组内需要进行 all-reduce 来同步
          每一层的激活值和梯度。

        - 每列的 dp_size 个设备组成一个数据并行组（DP group），处理不同
          的微批次数据。DP 组内需要进行 all-reduce 来同步梯度。

        每层的通信开销 = TP 组 all-reduce + DP 组 all-reduce。
        - TP 组 all-reduce：2 × (tp_size - 1) 步
        - DP 组 all-reduce：2 × (dp_size - 1) 步

    在分布式训练中的用途：
        Megatron-LM 使用此拓扑训练数十亿参数的大语言模型。
        TP 解决单 GPU 显存不足的问题，DP 提高吞吐量。
        理解此拓扑的通信代价有助于合理配置 TP 和 DP 的比例。

    Args:
        tp_size: 张量并行组大小（网格行宽）。
        dp_size: 数据并行组大小（网格列高）。

    Returns:
        包含以下字段的字典：
        - total_devices: 设备总数 tp_size × dp_size
        - tp_group_size: 张量并行组大小
        - dp_group_size: 数据并行组大小
        - communication_cost_per_layer: 每层通信代价（总步数）
        - tp_communication_steps: TP 组 all-reduce 步数
        - dp_communication_steps: DP 组 all-reduce 步数
        - description: 中文描述字符串
    """
    if tp_size < 1 or dp_size < 1:
        raise ValueError(
            f"tp_size 和 dp_size 必须 >= 1，当前值: tp_size={tp_size}, dp_size={dp_size}"
        )

    total_devices = tp_size * dp_size
    tp_steps = 2 * (tp_size - 1)
    dp_steps = 2 * (dp_size - 1)
    total_cost = tp_steps + dp_steps

    return {
        "total_devices": total_devices,
        "tp_group_size": tp_size,
        "dp_group_size": dp_size,
        "communication_cost_per_layer": total_cost,
        "tp_communication_steps": tp_steps,
        "dp_communication_steps": dp_steps,
        "description": (
            f"二维网格拓扑 (Megatron TP+DP)：{total_devices} 个设备排列为 "
            f"{tp_size}×{dp_size} 网格。共 {dp_size} 个 TP 组（每组 {tp_size} 卡，"
            f"同一样本的不同分片）和 {tp_size} 个 DP 组（每组 {dp_size} 卡，"
            f"不同样本的数据）。每层通信代价 = TP all-reduce({tp_steps}步) + "
            f"DP all-reduce({dp_steps}步) = {total_cost} 步。"
            f"TP 减小显存压力，DP 提高吞吐量，两者结合实现大规模高效训练。"
        ),
    }


def visualize_topology(world_size: int, topology_type: str) -> None:
    """打印指定拓扑的 ASCII 示意图。

    支持四种拓扑类型：
    - "ring": 环形拓扑，每个节点连接左右邻居，首尾相连成环。
    - "tree": 二叉树拓扑，rank 0 为根，每个内部节点有左右两个子节点。
    - "mesh_2x2": 2×2 二维网格拓扑。
    - "mesh_2x4": 2×4 二维网格拓扑。

    当 world_size 较小或不支持时回退到简易文本描述。

    Args:
        world_size: 进程/设备总数。
        topology_type: 拓扑类型，支持 "ring", "tree", "mesh_2x2", "mesh_2x4"。
    """
    print(f"\n{'='*60}")
    print(f"拓扑类型: {topology_type} (P={world_size})")
    print(f"{'='*60}")

    if topology_type == "ring":
        _visualize_ring(world_size)

    elif topology_type == "tree":
        _visualize_tree(world_size)

    elif topology_type.startswith("mesh_"):
        # 解析网格维度，如 "mesh_2x4" -> rows=2, cols=4
        try:
            parts = topology_type.replace("mesh_", "").split("x")
            rows = int(parts[0])
            cols = int(parts[1])
        except (IndexError, ValueError):
            print(f"[警告] 无法解析网格维度 '{topology_type}'，期望格式: mesh_RxC")
            return
        _visualize_mesh(rows, cols, world_size)

    else:
        print(f"[警告] 不支持的拓扑类型: '{topology_type}'")
        print(f"  支持的类型: ring, tree, mesh_2x2, mesh_2x4")


def _visualize_ring(world_size: int) -> None:
    """打印环形拓扑的 ASCII 图。"""
    if world_size <= 1:
        print("[0]")
        return

    if world_size <= 4:
        # 小型环：四角排列
        print("  0 ---> 1")
        print("  ^       |")
        print("  |       v")
        print("  3 <--- 2")
        return

    # 大型环：水平排列
    top = " +" + "--" * (world_size - 1) + "-+"
    print(top)
    line = " |"
    for i in range(world_size):
        line += f" {i}"
    line += " |"
    print(line)
    bottom = " +" + "--" * (world_size - 1) + "-+"
    print(bottom)


def _visualize_tree(world_size: int) -> None:
    """打印二叉树拓扑的 ASCII 图。"""
    if world_size <= 1:
        print("[0]")
        return

    height = math.ceil(math.log2(world_size))

    # 逐层构建节点行和连接行
    # 用层级信息递归构建每一层的节点列表
    levels = []
    node_id = 0

    num_levels = height + 1  # 树高为边数，层数为 height+1
    for level in range(num_levels):
        max_nodes = 2 ** level
        level_nodes = []
        for _ in range(max_nodes):
            if node_id < world_size:
                level_nodes.append(str(node_id))
                node_id += 1
            else:
                level_nodes.append("")
        levels.append(level_nodes)

    # 最大层宽度（最深层的节点位置）
    max_leaves = 2 ** height
    total_width = max_leaves * 4  # 每个叶子占 4 字符宽度

    for level_idx, level_nodes in enumerate(levels):
        nodes_in_level = 2 ** level_idx
        spacing = total_width // nodes_in_level
        line = ""
        for i, node in enumerate(level_nodes):
            pos = i * spacing + spacing // 2
            # 简单居中对齐
            pad = pos - len(line)
            line += " " * max(0, pad - len(node) // 2)
            line += node if node else " "
        print(line)

        # 打印连接线（除最后一层外）
        if level_idx < len(levels) - 1:
            next_nodes = levels[level_idx + 1]
            conn_line = ""
            for i, node in enumerate(level_nodes):
                pos = i * spacing + spacing // 2
                pad = pos - len(conn_line)
                conn_line += " " * max(0, pad)
                if node:
                    left_child_idx = i * 2
                    right_child_idx = i * 2 + 1
                    has_left = left_child_idx < len(next_nodes) and next_nodes[left_child_idx]
                    has_right = right_child_idx < len(next_nodes) and next_nodes[right_child_idx]
                    if has_left or has_right:
                        conn_line += "/"
                    if has_left and has_right:
                        conn_line += " \\"
                    elif has_right:
                        conn_line += "\\"
            print(conn_line)


def _visualize_mesh(rows: int, cols: int, world_size: int) -> None:
    """打印二维网格拓扑的 ASCII 图。"""
    actual = rows * cols
    if actual < world_size:
        print(f"[警告] 网格 {rows}×{cols}={actual} < P={world_size}，部分节点不显示")
    elif actual > world_size:
        print(f"[警告] 网格 {rows}×{cols}={actual} > P={world_size}，多余位置留空")

    # 确定每个节点号需要的最大宽度
    max_width = max(len(str(rows * cols - 1)), len(str(world_size - 1))) + 1

    for r in range(rows):
        # 节点行
        node_line = ""
        for c in range(cols):
            node_id = r * cols + c
            if node_id < world_size:
                node_line += f"{node_id:^{max_width}}"
            else:
                node_line += f"{'':^{max_width}}"
            if c < cols - 1:
                node_line += "--"
        print(node_line)

        # 垂直连接线（除最后一行外）
        if r < rows - 1:
            conn_line = ""
            for c in range(cols):
                conn_line += f"{'|':^{max_width}}"
                if c < cols - 1:
                    conn_line += "  "
            print(conn_line)


if __name__ == "__main__":
    for size in [4, 8]:
        ring = analyze_ring_topology(size)
        print(f"Ring P={size}: {ring['total_steps']} steps, {ring['per_step_data']} data/step")

        tree = analyze_tree_topology(size)
        print(f"Tree P={size}: height={tree['tree_height']}, broadcast steps={tree['total_steps_for_broadcast']}")

    mesh = analyze_mesh_topology(tp_size=4, dp_size=2)
    print(f"Mesh 4x2: {mesh['total_devices']} devices, cost={mesh['communication_cost_per_layer']}")

    print("\nTopology visualizations:")
    for t in ["ring", "tree", "mesh_2x4"]:
        visualize_topology(8, t)
