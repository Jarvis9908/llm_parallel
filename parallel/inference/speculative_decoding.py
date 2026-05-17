"""推测解码 (Speculative Decoding)：用小模型快速生成候选 token 序列，再用大模型并行验证，加速自回归推理。"""
import torch


def draft_generate(
    draft_model, prompt: torch.LongTensor, n_candidates: int = 5
) -> torch.LongTensor:
    """
    使用草稿模型快速生成候选 token 序列。

    草稿模型参数量小、推理速度快，按自回归方式逐 token 生成一个
    候选序列，供后续目标模型进行并行验证。

    Args:
        draft_model: 小参数量的草稿模型（如 n=0.1B 的 Student 模型）
        prompt: 输入 prompt token 序列，形状 (batch, seq_len)
        n_candidates: 需要生成的候选 token 数量

    Returns:
        torch.LongTensor: 生成的候选 token 序列，形状 (batch, n_candidates)
    """
    draft_model.eval()
    generated = prompt.clone()
    with torch.no_grad():
        for _ in range(n_candidates):
            logits = draft_model(generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated[:, -n_candidates:]


def target_verify(
    target_model,
    prompt: torch.LongTensor,
    candidates: torch.LongTensor,
) -> tuple[torch.LongTensor, int]:
    """
    用大模型并行验证候选 token 序列。

    将 prompt 与候选序列拼接后一次前向传播，大模型可同时计算每个位置
    的 logits。然后逐位置比对草稿模型的生成结果与大模型的预测结果：
    如果匹配则接受，一旦不匹配则停止（按需回退）。
    当前实现为简化版，直接接受所有候选。

    Args:
        target_model: 大参数量的目标模型
        prompt: 输入 prompt token 序列，形状 (batch, seq_len)
        candidates: 草稿模型生成的候选 token，形状 (batch, n_candidates)

    Returns:
        (accepted_tokens, n_accepted):
            accepted_tokens: 被接受的 token 序列
            n_accepted: 被接受的 token 数量
    """
    target_model.eval()
    full_input = torch.cat([prompt, candidates], dim=1)
    with torch.no_grad():
        logits = target_model(full_input)
    # 简化实现：接受所有候选 token（实际场景需逐位置比对验证）
    return candidates, candidates.shape[1]


def speedup_analysis(
    draft_time_ms: float, target_time_ms: float, n_accepted: int, n_candidates: int
) -> float:
    """
    计算推测解码相对于普通自回归解码的加速比。

    推测解码的总耗时 = 草稿模型生成时间 + 目标模型验证时间。
    基准耗时 = 目标模型单步时间 * 接受的 token 数（即不用推测解码时
    需要逐 token 生成的耗时）。

    Args:
        draft_time_ms: 草稿模型生成候选序列的耗时（毫秒）
        target_time_ms: 目标模型单次前向验证的耗时（毫秒）
        n_accepted: 被目标模型接受的 token 数量
        n_candidates: 草稿模型生成的候选 token 总数

    Returns:
        float: 加速比，>1 表示推测解码更快，<1 表示反而变慢
    """
    # 每次 target forward 验证 n_candidates 个候选 token
    speculative_time = draft_time_ms + target_time_ms
    # 不用推测解码时需要逐 token 调用 target model 的累计时间
    baseline_time = target_time_ms * n_accepted
    return baseline_time / speculative_time


if __name__ == "__main__":
    print("推测解码演示：")

    # 模拟场景：draft 模型生成 5 个候选，target 模型接受 4 个
    n_candidates = 5
    n_accepted = 4

    # 假设 draft 前向 5 次共耗时 10ms，target 单次前向耗时 50ms
    draft_time_ms = 10.0
    target_time_ms = 50.0

    speedup = speedup_analysis(draft_time_ms, target_time_ms, n_accepted, n_candidates)
    print(f"  Draft 模型耗时: {draft_time_ms} ms")
    print(f"  Target 模型单次耗时: {target_time_ms} ms")
    print(f"  候选 token 数: {n_candidates}, 接受 token 数: {n_accepted}")
    print(f"  推测解码加速比: {speedup:.2f}x")

    # 分析：不同接受率下的加速效果
    print("\n不同接受率下的加速比：")
    for accepted in range(1, n_candidates + 1):
        sp = speedup_analysis(draft_time_ms, target_time_ms, accepted, n_candidates)
        acceptance_rate = accepted / n_candidates
        print(f"  接受率 {acceptance_rate:.0%} ({accepted}/{n_candidates}): {sp:.2f}x")
