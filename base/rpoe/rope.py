import math
import torch  # 用torch实现（面试中Transformer相关手撕优先用torch，更贴合实际）

def rotate_half(x):
    """
    核心辅助函数：将向量后半部分维度旋转（x1, x2）→（-x2, x1）
    对应RoPE旋转矩阵的核心操作，面试必须讲清这个函数的作用
    """
    # 拆分维度：假设输入shape是 [seq_len, dim]，按最后一维切分
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    # 旋转操作：(x1, x2) → (-x2, x1)，拼接后返回
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, pos_emb):
    """
    核心函数：对Query和Key应用RoPE旋转位置编码
    :param q: Query向量，shape [seq_len, dim]
    :param k: Key向量，shape [seq_len, dim]
    :param pos_emb: 预计算的位置角频率，shape [seq_len, dim]
    :return: 加了位置编码的q、k
    """
    # 步骤1：按RoPE公式计算cos和sin
    q_rot = q * pos_emb.cos() + rotate_half(q) * pos_emb.sin()
    k_rot = k * pos_emb.cos() + rotate_half(k) * pos_emb.sin()
    return q_rot, k_rot

def precompute_freqs_cis(dim, max_seq_len):
    """
    预计算位置角频率（面试简化版，核心逻辑）
    :param dim: 向量维度（必须是偶数）
    :param max_seq_len: 最大序列长度
    :return: 位置角频率，shape [max_seq_len, dim]
    """
    # 步骤1：计算θ_i = 10000^(-2(i-1)/dim)，i为维度索引（RoPE核心公式）
    theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2) / dim))  # [dim//2]
    # 步骤2：生成位置索引 0,1,...,max_seq_len-1
    seq_idx = torch.arange(max_seq_len)  # [max_seq_len]
    # 步骤3：位置×角频率，广播为 [max_seq_len, dim//2]
    freqs = torch.outer(seq_idx, theta)
    # 步骤4：扩展到完整维度（两两重复，适配rotate_half的分组）
    freqs = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
    return freqs

# ------------------- 面试演示用测试代码 -------------------
if __name__ == "__main__":
    # 1. 模拟参数（面试中用简单参数，易计算验证）
    dim = 4  # 向量维度（必须偶数，简化计算）
    max_seq_len = 3  # 序列长度：0,1,2
    seq_len = 3
    
    # 2. 预计算位置角频率
    freqs = precompute_freqs_cis(dim, max_seq_len)
    
    # 3. 模拟Query和Key向量（shape [seq_len, dim]）
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]])
    k = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                      [0.5, 0.6, 0.7, 0.8],
                      [0.9, 1.0, 1.1, 1.2]])
    
    # 4. 应用RoPE
    q_rot, k_rot = apply_rotary_pos_emb(q, k, freqs[:seq_len])
    
    # 5. 输出结果（面试中可简单解释结果含义）
    print("RoPE后的Query:\n", q_rot)
    print("RoPE后的Key:\n", k_rot)