"""
KL 散度的三种写法: 精确解 + 两种蒙特卡洛估计 (k1 / k3)

约定方向: KL(p || q), 样本从 p 采样 (RLHF 中 p = 策略, q = 参考模型)
记 r = q(x) / p(x), 则 log r = log q - log p

    精确: KL = Σ_x p(x) * (log p - log q)          需要完整分布, 词表维度求和
    k1  = -log r = log p - log q                   无偏, 但单样本可正可负
    k3  = r - log r - 1                            无偏, 且恒 >= 0

k1 无偏: E_p[log p - log q] 就是 KL 的定义
k3 无偏: E_p[r] = Σ p * q/p = 1, 故 E[r - 1] = 0, 减去它不改变期望 (控制变量法)
k3 非负: x - log x - 1 >= 0 对任意 x > 0 成立 (等号仅在 x = 1 处)
(还有 k2 = 0.5 * (log r)^2, 非负但有偏, 实践中被 k3 取代)

易踩的坑: k3 方差更小只在 p ≈ q 时成立 (RLHF 里策略被约束在参考模型附近, 正是这种情形)。
两个分布差很远时, r 的 exp 会在尾部爆炸, k3 方差反而远大于 k1 —— 见下方 demo 的对照输出。
"""

import torch
import torch.nn.functional as F


# ============ 1. 精确 KL: 已知完整分布, 在词表维度上求和 ============
def kl_exact(logits_p, logits_q):
    """输入 logits, shape [..., V]; 返回 shape [...]"""
    # log_softmax 内部先减 max 再求和, 比 log(softmax(x)) 数值稳定
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


# ============ 2. k1 估计器: 只需采样点的 log 概率 ============
def kl_k1(logp, logq):
    """logp/logq 为被采样 token 的对数概率, 任意 shape (如 [B, T])"""
    return logp - logq


# ============ 3. k3 估计器: 加控制变量 (r - 1), 降方差且保证非负 ============
def kl_k3(logp, logq):
    log_ratio = logq - logp                    # log r
    return log_ratio.exp() - log_ratio - 1     # r - log r - 1


if __name__ == "__main__":
    torch.manual_seed(0)
    V, N = 8, 100_000
    logits_p = torch.randn(V) * 1.5                # 策略分布
    logp_full = F.log_softmax(logits_p, dim=-1)

    # 对照两种情形: 参考模型离策略多远, 直接决定 k3 到底降不降方差
    cases = [
        ("p ≈ q  (RLHF 常态)", logits_p + torch.randn(V) * 0.2),
        ("q 与 p 无关 (r 尾部爆炸)", torch.randn(V) * 1.5),
    ]
    for desc, logits_q in cases:
        logq_full = F.log_softmax(logits_q, dim=-1)

        # 从 p 采样 N 次, 只取采样 token 的 log 概率 (模拟拿不到完整分布的场景)
        idx = torch.multinomial(logp_full.exp(), num_samples=N, replacement=True)
        logp, logq = logp_full[idx], logq_full[idx]
        k1, k3 = kl_k1(logp, logq), kl_k3(logp, logq)

        print(f"\n=== {desc} ===")
        print(f"精确 KL(p||q): {kl_exact(logits_p, logits_q).item():.4f}")
        print(f"k1 -> 均值 {k1.mean():.4f} | 方差 {k1.var():8.4f} | 负值占比 {(k1 < 0).float().mean():.1%}")
        print(f"k3 -> 均值 {k3.mean():.4f} | 方差 {k3.var():8.4f} | 负值占比 {(k3 < 0).float().mean():.1%}")

