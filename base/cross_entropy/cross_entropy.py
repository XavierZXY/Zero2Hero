"""
交叉熵损失: 二分类 (BCE) 与多分类 (CE)

    BCE = -[y*log(p) + (1-y)*log(1-p)]          p = sigmoid(logit),   标量输出
    CE  = -Σ_c y_c * log(p_c) = -log(p_y)        p = softmax(logits), C 维输出

两者是同一个东西: BCE 就是 CE 在 C=2 且用 sigmoid 参数化时的特例。
多分类里 y 是 one-hot, 求和只剩真实类别那一项, 所以 CE 退化成"取出真实类别的对数概率"
—— 实现上不需要真的构造 one-hot 矩阵, 用 gather 按下标取即可。

数值稳定是本题的考点:
    softmax 先减 max (softmax 平移不变), 否则 exp(大数) 上溢成 inf
    log 概率用 logits - logsumexp(logits) 一步算出 (即 log_softmax),
    而不是 log(softmax(x)) —— 后者在概率下溢到 0 时会得到 -inf, 见 demo 3

梯度结论 (面试高频): dCE/dlogits = p - y, 形式极简, 这正是 softmax + CE 配对的原因。
"""

import torch
import torch.nn.functional as F
import numpy as np


# ============ 1. 二分类交叉熵 ============
def binary_cross_entropy_loss(y_pred, y_true):
    # 1. Sigmoid 转换: 将模型输出映射到 [0,1] 概率区间
    y_pred = 1 / (1 + torch.exp(-y_pred))

    # 2. 数值稳定: 限制预测值范围, 避免 log(0) 导致梯度爆炸/NaN
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, min=epsilon, max=1.0 - epsilon)

    # 3. 计算交叉熵损失: 逐样本计算后求平均
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()


class BinaryClassifier(torch.nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer = torch.nn.Linear(2, 1)  # 输入维度 2, 输出维度 1 (二分类)

    def forward(self, x):
        return self.layer(x)


# ============ 2. 多分类交叉熵 ============
def softmax(logits):
    """手写 softmax, 输入 [..., C]; 减 max 是为了防 exp 上溢, 不改变结果"""
    logits = logits - logits.max(dim=-1, keepdim=True).values
    exp = logits.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def cross_entropy_loss(logits, targets):
    """logits [N, C] 未过 softmax; targets [N] 为类别下标; 返回标量

    等价于 F.cross_entropy(logits, targets), 即 log_softmax + NLL 两步合一
    """
    # log_softmax: logsumexp 内部已做减 max 处理, 比 log(softmax(x)) 稳定
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)

    # one-hot 求和退化为按真实类别下标取值, 无需构造 [N, C] 的 one-hot 矩阵
    nll = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    return nll.mean()


class MultiClassifier(torch.nn.Module):
    def __init__(self, in_dim=4, num_classes=3):
        super(MultiClassifier, self).__init__()
        self.layer = torch.nn.Linear(in_dim, num_classes)  # 输出 C 个 logits, 不接 softmax

    def forward(self, x):
        return self.layer(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- demo 1: 二分类 ----
    model = BinaryClassifier()
    features = np.array([[0.2, 0.4], [0.6, 0.8], [0.1, 0.3], [0.5, 0.7]])
    labels = np.array([1, 0, 1, 0])

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # 适配模型输出维度

    outputs = model(features)
    loss = binary_cross_entropy_loss(outputs, labels)
    print(f"二分类交叉熵损失: {loss.item():.6f}")
    print(f"对照 F.binary_cross_entropy_with_logits: "
          f"{F.binary_cross_entropy_with_logits(outputs, labels).item():.6f}")

    # ---- demo 2: 多分类, 与官方实现对齐 + 验证梯度 = (p - y) / N ----
    N, C = 4, 3
    mc_model = MultiClassifier(in_dim=4, num_classes=C)
    x = torch.randn(N, 4)
    y = torch.tensor([0, 2, 1, 2])

    logits = mc_model(x)
    logits.retain_grad()
    my_loss = cross_entropy_loss(logits, y)
    my_loss.backward()

    print(f"\n多分类交叉熵损失: {my_loss.item():.6f}")
    print(f"对照 F.cross_entropy: {F.cross_entropy(logits, y).item():.6f}")

    # 梯度理论值: (softmax(logits) - one_hot(y)) / N, 除以 N 是因为 loss 取了 mean
    grad_expect = (softmax(logits) - F.one_hot(y, C).float()) / N
    print(f"梯度是否等于 (p - y)/N: {torch.allclose(logits.grad, grad_expect, atol=1e-6)}")
    print(f"预测概率 (手写 softmax):\n{softmax(logits).detach().round(decimals=3)}")

    # ---- demo 3: 为什么必须用 log_softmax 而不是 log(softmax(x)) ----
    extreme = torch.tensor([[1000.0, 0.0, -1000.0]])   # 量级悬殊的 logits
    naive = softmax(extreme).log()                      # 概率下溢到 0 -> log(0) = -inf
    stable = extreme - extreme.logsumexp(dim=-1, keepdim=True)
    print(f"\nlog(softmax(x)) : {naive.tolist()}")
    print(f"log_softmax(x)  : {stable.tolist()}")
