import math

import torch
import torch.nn as nn


class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // self.nums_head

        # 投影层保持不变
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(0.1)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # 初始化KV缓存（None表示无缓存）
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        """重置KV缓存，用于新序列的开始"""
        self.cache_k = None
        self.cache_v = None

    def forward(self, X, attention_mask=None, use_cache=False):
        # X shape: (batch, seq_len, hidden_dim)
        # attention_mask shape: (batch, seq_len)
        # use_cache: 是否使用KV缓存（推理阶段为True，训练阶段为False）
        batch_size, seq_len, _ = X.size()

        # 1. 计算当前token的Q/K/V
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # 2. 重塑Q/K/V的形状: (batch, num_head, seq_len, head_dim)
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)

        # 3. 处理KV Cache
        if use_cache:
            # 推理阶段：复用历史K/V，拼接新的K/V
            if self.cache_k is not None and self.cache_v is not None:
                # 拼接历史缓存和当前K/V (batch, num_head, total_seq_len, head_dim)
                k_state = torch.cat([self.cache_k, k_state], dim=2)
                v_state = torch.cat([self.cache_v, v_state], dim=2)
            
            # 更新缓存
            self.cache_k = k_state
            self.cache_v = v_state
        else:
            # 训练阶段：清空缓存（防止干扰）
            self.reset_cache()

        # 4. 计算注意力权重: (batch, num_head, seq_len, total_seq_len)
        attn_weight = q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim)

        # 5. 处理attention mask
        if attention_mask is not None:
            # 适配mask形状: (batch, 1, seq_len, total_seq_len)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
                # 扩展到匹配attn_weight的最后一维
                total_seq_len = k_state.size(2)
                attention_mask = attention_mask.expand(-1, -1, seq_len, total_seq_len)
            attn_weight = attn_weight.masked_fill(attention_mask == 0, float("-1e20"))

        # 6. 计算注意力输出
        attn_weight = self.attn_dropout(torch.softmax(attn_weight, dim=-1))
        output_mid = attn_weight @ v_state  # (batch, num_head, seq_len, head_dim)
        
        # 7. 重塑输出形状
        output_mid = output_mid.transpose(1, 2).contiguous()  # (batch, seq_len, num_head, head_dim)
        output = output_mid.view(batch_size, seq_len, -1)      # (batch, seq_len, hidden_dim)
        ret = self.o_proj(output)

        return ret


def main():
    # 测试1: 训练模式（不使用KV Cache）
    print("===== 训练模式（无KV Cache） =====")
    attention_mask = torch.tensor([[1, 1], [1, 1], [1, 1]])  # 3个样本，每个样本2个token
    x = torch.rand(3, 2, 128)  # (batch=3, seq_len=2, hidden_dim=128)
    net = MultiHeadAttentionWithKVCache(128, 8)
    output = net(x, attention_mask, use_cache=False)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 预期: (3, 2, 128)

    # 测试2: 推理模式（使用KV Cache，模拟逐token生成）
    print("\n===== 推理模式（有KV Cache） =====")
    net.reset_cache()  # 重置缓存
    batch_size = 3
    hidden_dim = 128

    # 第一步：输入第一个token（seq_len=1）
    x_step1 = torch.rand(batch_size, 1, hidden_dim)  # (3, 1, 128)
    mask_step1 = torch.ones(batch_size, 1)           # (3, 1)
    output_step1 = net(x_step1, mask_step1, use_cache=True)
    print(f"第一步 - 输入形状: {x_step1.shape}, 输出形状: {output_step1.shape}")  # (3,1,128)
    print(f"第一步 - KV缓存形状: K={net.cache_k.shape}, V={net.cache_v.shape}")   # (3,8,1,16)

    # 第二步：输入第二个token（seq_len=1），复用第一步的KV缓存
    x_step2 = torch.rand(batch_size, 1, hidden_dim)  # (3, 1, 128)
    mask_step2 = torch.ones(batch_size, 2)           # (3, 2) （覆盖前2个token）
    output_step2 = net(x_step2, mask_step2, use_cache=True)
    print(f"第二步 - 输入形状: {x_step2.shape}, 输出形状: {output_step2.shape}")  # (3,1,128)
    print(f"第二步 - KV缓存形状: K={net.cache_k.shape}, V={net.cache_v.shape}")   # (3,8,2,16)

    # 重置缓存，准备下一个序列
    net.reset_cache()
    print(f"重置缓存后: K={net.cache_k}, V={net.cache_v}")


if __name__ == "__main__":
    main()