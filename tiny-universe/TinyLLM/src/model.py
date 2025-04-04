import inspect
import math
import struct
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    # 自定义超参数
    dim: int = 288  # 模型维度
    n_layers: int = 6  # Transformer层数
    n_heads: int = 6  # 注意力机制的头数
    n_kv_heads: Optional[int] = 6  # 键/值头数，如果未指定，则默认为n_heads
    vocab_size: int = 32000  # 词汇表大小
    hidden_dim: Optional[int] = (
        None  # 隐藏层维度，如果未指定，则使用其他规则确定
    )
    multiple_of: int = 32  # MLP隐藏层大小是这个数的倍数
    norm_eps: float = 1e-5  # 归一化层的epsilon值
    max_seq_len: int = 256  # 最大序列长度
    dropout: float = 0.0  # 丢弃率


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # forward函数调用了_norm函数，实现了RMSNorm的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# 获得旋转嵌入的实部和虚部
# Note: 此处的dim应为dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    # 生成了一个从0到end的序列
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


# 此函数的作用是将freqs_cis调整为与x的形状相同，以便能够与x进行广播操作
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度数
    ndim = x.ndim
    # 断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小，序列长度，键值头数，隐藏层维度
    bs, slen, n_kv_heads, head_dim = x.shape
    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x
    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :]  # 在第3维度上增加一个维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 扩展张量
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重塑张量
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = (
            args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        )
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理
        model_parallel_size = 1
        # 本地计算头数
        self.local_heads = args.n_heads // model_parallel_size
        # 本地键/值头数
        self.local_n_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键/值
        self.n_rep = self.local_heads // self.local_n_kv_heads
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        # 输出权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 检查是否使用Flash Attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full(
                (1, 1, args.max_seq_len, args.max_seq_len), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape
        # 计算Q,K,V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 将Q,K,V分割为多个头
        xq = xq.view(bsz, seqlen, self.local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        # 应用旋转位置嵌入RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        # 重复键/值，以便在多个头之间共享，减少K,V的计算量，多个Q之间共享同一个K,V
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        # 将头作为批次处理
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # 根据是否支持FLash Attention选择不同的注意力机制
        if self.flash:
            # 使用Flash Attention。
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 使用手动实现的注意力机制。
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # 将多头注意力的结果连接起来
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class MLP(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, multiple_of: int, dropout: float
    ):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * (
                (hidden_dim + multiple_of - 1) // multiple_of
            )
        # 第一层线性变化，输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads
        # 定义输入的维度
        self.dim = args.dim
        # 定义每一个头的维度
        self.head_dim = args.dim // args.n_heads
        # 定义LLaMa2Attention对象，用于进行多头注意力的计算
        self.attention = Attention(args)
        # 定义MLP对象，用于进行多层感知机的计算
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层的id
        self.layer_id = layer_id
        # 定义RMSNorm对象，用于进行RMSNorm的计算
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight
        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers)
                )

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None

    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 前向传播函数
        _bsz, seqlen = tokens.shape
        # 获取词嵌入
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        # 通过Decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type
    ):
        # 获取所有需要更新的参数
        param_dict = {
            pn: p for pn, p in self.named_parameters() if p.requires_grad
        }
        # 将参数分为需要权重衰减和不需要权重衰减的两组
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # 打印参数数量信息
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # 根据设备类型选择使用标准 AdamW 或其融合版本
        fused_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """估计模型的 FLOPs 利用率 (MFU) 单位：A100 bfloat16 的峰值 FLOPS"""
        # 计算每次迭代的 FLOPs 数量（参考 PaLM 论文的附录 B）
        # PaLM: Scaling Language Modeling with Pathways: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.args
        L, H, Q, T = (
            cfg.n_layers,
            cfg.n_heads,
            cfg.dim // cfg.n_heads,
            cfg.max_seq_len,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # 将 FLOPs 吞吐量表示为 A100 bfloat16 峰值 FLOPS 的比例
        flops_achieved = flops_per_iter * (1.0 / dt)  # 每秒计算的 FLOPs
        flops_promised = 312e12  # A100 GPU bfloat16 的峰值 FLOPS 为 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """给定输入序列idx(bs, seq_len),通过多次生成新的token来完成序列
        在model.eval()模式下运行，效率较低的采样版本，没有使用k/v cache

        Args:
            idx (_type_): _description_
            max_new_tokens (_type_): _description_
            temperature (float, optional): _description_. Defaults to 1.0.
            top_k (_type_, optional): _description_. Defaults to None.
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大的长度
            idx_cond = (
                idx
                if idx.size(1) <= self.args.max_seq_len
                else idx[:, -self.args.max_seq_len :]
            )
            # 前向传播获取序列中的最后一个位置的logits
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]  # 保留最后一个位置的logits
            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    args = ModelArgs()
    # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
    x = torch.randint(0, 32000, (1, 50))  # [bs, seq_len]
    # 实例化LLaMA2Model
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters:", num_params)

    out = model(x)
    print(out.shape)  # [batch_size, 1, vocab_size]
