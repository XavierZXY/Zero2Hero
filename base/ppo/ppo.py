"""
PPO 核心两件套: 截断重要性采样目标 (clipped surrogate) + GAE 优势估计

    ratio_t = π_θ(a|s) / π_old(a|s) = exp(logp - logp_old)
    L_clip  = -E[ min( ratio * A,  clip(ratio, 1-ε, 1+ε) * A ) ]

为什么用 ratio 而不是 logp 之差:
    数据由 π_old 采集, 更新 π_θ 属于离策略, 需要重要性采样权重修正分布偏移。
    ratio 在 θ = θ_old 处等于 1, 此时 L_clip 的梯度正好退化成策略梯度 ∇logp * A。

为什么取 min (而不是直接 clip):
    min 让目标成为未截断目标的悲观下界 —— 只在"策略朝有利方向走过头"时砍掉梯度,
    朝不利方向走多远都保留梯度, 保证坏更新总能被拉回来。见 demo 2 的四象限验证。

GAE: A_t = Σ_k (γλ)^k δ_{t+k},   δ_t = r_t + γV(s_{t+1})(1-done_t) - V(s_t)
    λ=0 -> A_t = δ_t         纯 TD, 低方差高偏差 (完全信任 V)
    λ=1 -> A_t = G_t - V(s_t) 纯 MC, 无偏但高方差
    λ∈(0,1) 在两者间插值, 实践常取 0.95。demo 1 用这两个极限做正确性验证。

(完整 PPO 目标还有 value loss 与 entropy bonus:
 L = L_clip + vf_coef * MSE(V, returns) - ent_coef * H(π), 后两项不是本题重点)
"""

import torch


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """倒序递推计算 GAE, 输入均为 [T] 的一维张量, last_value 为 s_T 的 bootstrap 值

    dones[t] = 1 表示 s_{t+1} 是终止态, 此时既不 bootstrap 也不向前传递优势
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values          # V 的回归目标 = 优势 + 基线
    return advantages, returns


def ppo_policy_loss(logp, logp_old, advantages, clip_eps=0.2):
    """logp/logp_old/advantages 同形状 (如 [N] 或 [B, T]); 返回标量

    logp_old 来自采样时的旧策略, 必须是常量 (detach), 否则 ratio 恒为 1
    """
    ratio = (logp - logp_old).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()   # 取负号: 最大化目标 = 最小化损失


if __name__ == "__main__":
    torch.manual_seed(0)
    gamma = 0.99

    # ---- demo 1: 用 λ 的两个极限验证 GAE 实现正确 ----
    T = 5
    rewards = torch.randn(T)
    values = torch.randn(T)
    dones = torch.zeros(T)                 # 整段轨迹未终止, 末尾走 bootstrap
    last_value = torch.tensor(0.5)

    adv_td, _ = compute_gae(rewards, values, dones, last_value, gamma, lam=0.0)
    adv_mc, _ = compute_gae(rewards, values, dones, last_value, gamma, lam=1.0)

    # λ=0 的理论值: δ_t
    next_values = torch.cat([values[1:], last_value.view(1)])
    delta = rewards + gamma * next_values - values

    # λ=1 的理论值: 折扣回报 G_t (含 bootstrap) 减去基线 V_t
    returns_mc = torch.zeros(T)
    g = last_value
    for t in reversed(range(T)):
        g = rewards[t] + gamma * g
        returns_mc[t] = g

    print(f"GAE(λ=0) == TD 残差 δ_t      : {torch.allclose(adv_td, delta, atol=1e-6)}")
    print(f"GAE(λ=1) == MC 回报 G_t - V_t: {torch.allclose(adv_mc, returns_mc - values, atol=1e-5)}")

    # ---- demo 2: clip 的四象限行为, 看梯度何时被砍成 0 ----
    print(f"\n{'优势 A':>8} {'ratio':>7} {'loss':>9} {'d(loss)/d(logp)':>17}   是否截断")
    for adv_val, ratio_val in [(1.0, 1.5), (1.0, 0.5), (-1.0, 1.5), (-1.0, 0.5)]:
        # 令 logp_old = 0, 则 logp = log(ratio)
        logp = torch.tensor([ratio_val]).log().requires_grad_(True)
        logp_old = torch.zeros(1)
        adv = torch.tensor([adv_val])

        loss = ppo_policy_loss(logp, logp_old, adv, clip_eps=0.2)
        loss.backward()
        grad = logp.grad.item()
        print(f"{adv_val:>8.1f} {ratio_val:>7.1f} {loss.item():>9.4f} {grad:>17.4f}"
              f"   {'是 (梯度归零)' if grad == 0 else '否'}")

    # ---- demo 3: 优势归一化, 稳定训练的常用技巧 ----
    adv = torch.randn(8) * 10 + 5
    normed = (adv - adv.mean()) / (adv.std() + 1e-8)
    print(f"\n归一化前 均值 {adv.mean():.4f} 标准差 {adv.std():.4f}")
    print(f"归一化后 均值 {normed.mean():.4f} 标准差 {normed.std():.4f}")
