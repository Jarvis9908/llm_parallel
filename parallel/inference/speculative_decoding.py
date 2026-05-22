"""推测解码 (Speculative Decoding)：用小模型快速生成候选 token 序列，再用大模型并行验证，加速自回归推理。

直觉
----
小模型先猜，大模型验证——就像考试时先快速写出答案，再仔细检查。
小模型（draft model）参数少、推理快，逐 token 生成一个候选序列；
大模型（target model）参数多、精度高，一次前向传播就能并行验证
所有候选 token。接受正确的，拒绝错误的，从修正分布中采样替代 token。

数学
----
1. 验证过程：对于候选 token x_t，接受概率为
        p_accept = min(1, p_target(x_t) / p_draft(x_t))
   即大模型概率 / 小模型概率。若 p_target > p_draft，一定接受；
   若 p_target < p_draft，以概率比接受。

2. 修正分布采样：当拒绝 token x_t 时，从修正分布中采样替代 token：
        p_reject(x) ∝ max(0, p_target(x) - p_draft(x))
   这保证了最终输出严格服从大模型分布，即无损保证。

3. 加速比分析：
        speedup = n_accepted / (1 + t_draft / t_target)
   其中 n_accepted 是平均每次验证接受的 token 数，
   t_draft 是草稿模型生成时间，t_target 是目标模型验证时间。
   当接受率高且草稿模型快时，加速比可达 2-3x。

4. 无损保证：推测解码的输出分布与纯大模型自回归解码完全一致，
   因为拒绝时的修正采样补偿了接受概率的差异。

代码流程
--------
1. ``draft_generate`` —— 草稿模型生成候选 token 序列
2. ``target_verify`` —— 目标模型并行验证候选 token
3. ``speedup_analysis`` —— 加速比分析
"""
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
    draft_model=None,
    temperature: float = 1.0,
) -> tuple[torch.LongTensor, int]:
    """
    用大模型并行验证候选 token 序列（完整实现）。

    直觉：大模型一次性看完所有候选 token，逐个检查"我会不会也生成这个 token"。
    猜对的保留，猜错的从大模型的分布重新采样。

    数学（接受-拒绝采样）：
        对每个候选位置 t：
            1. 计算接受概率：p_accept = min(1, p_target(x_t) / p_draft(x_t))
            2. 以概率 p_accept 决定是否接受
            3. 如果接受：继续验证下一个位置
            4. 如果拒绝：
               a. 从修正分布采样：p_corrected(x) ∝ max(0, p_target(x) - p_draft(x))
               b. 返回已接受的 token + 修正采样的 token

        无损保证：最终输出分布与目标模型自回归分布完全相同。

    Args:
        target_model: 大参数量的目标模型
        prompt: 输入 prompt token 序列，形状 (batch, seq_len)
        candidates: 草稿模型生成的候选 token，形状 (batch, n_candidates)
        draft_model: 草稿模型（用于计算接受概率），如果为 None 则接受所有候选
        temperature: 采样温度

    Returns:
        (accepted_tokens, n_accepted):
            accepted_tokens: 被接受的 token 序列 + 修正采样 token，形状 (batch, variable)
            n_accepted: 被接受的候选 token 数量（不含修正采样 token）
    """
    target_model.eval()
    batch_size = prompt.shape[0]
    n_candidates = candidates.shape[1]

    # 拼接 prompt 和候选序列，一次前向传播
    full_input = torch.cat([prompt, candidates], dim=1)
    with torch.no_grad():
        logits = target_model(full_input)

    # 如果没有草稿模型，简化为接受所有候选
    if draft_model is None:
        return candidates, n_candidates

    # 计算草稿模型的概率
    draft_input = torch.cat([prompt, candidates[:, :-1]], dim=1)
    with torch.no_grad():
        draft_logits = draft_model(draft_input)

    # 逐位置验证
    accepted_list = []
    n_accepted = 0

    for t in range(n_candidates):
        # 目标模型在位置 prompt_len + t - 1 的预测（预测第 t 个候选）
        target_probs = torch.softmax(logits[:, prompt.shape[1] + t - 1] / temperature, dim=-1)
        draft_probs = torch.softmax(draft_logits[:, prompt.shape[1] + t - 1] / temperature, dim=-1)

        # 候选 token
        candidate_token = candidates[:, t]  # (batch,)

        # 接受概率
        p_target = target_probs.gather(1, candidate_token.unsqueeze(1)).squeeze(1)
        p_draft = draft_probs.gather(1, candidate_token.unsqueeze(1)).squeeze(1)
        p_accept = torch.min(torch.ones_like(p_target), p_target / (p_draft + 1e-10))

        # 采样决定是否接受
        accept = torch.rand_like(p_accept) < p_accept

        if accept.all():
            accepted_list.append(candidate_token)
            n_accepted += 1
        else:
            # 至少一个 batch 元素拒绝，记录已接受的 token 并从修正分布采样
            accepted_list.append(candidate_token * accept.long())

            # 修正分布采样：p_corrected ∝ max(0, p_target - p_draft)
            corrected_probs = torch.clamp(target_probs - draft_probs, min=0)
            corrected_probs = corrected_probs / (corrected_probs.sum(dim=-1, keepdim=True) + 1e-10)
            corrected_token = torch.multinomial(corrected_probs, 1).squeeze(1)
            # 在拒绝的位置使用修正采样的 token
            for b in range(batch_size):
                if not accept[b]:
                    accepted_list[-1][b] = corrected_token[b]
            n_accepted += accept.sum().item()
            break

    # 如果所有候选都被接受，额外从目标模型采样一个 token
    if len(accepted_list) == n_candidates:
        target_probs_last = torch.softmax(logits[:, -1] / temperature, dim=-1)
        extra_token = torch.multinomial(target_probs_last, 1).squeeze(1)
        accepted_list.append(extra_token)

    accepted_tokens = torch.stack(accepted_list, dim=1)
    return accepted_tokens, n_accepted


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
