import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------- 1. 定义基础组件 ----------------------
class Expert(nn.Module):
    """简单的专家网络（MLP）"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class GatingNetwork(nn.Module):
    """门控网络：输出每个样本分配给各专家的权重（softmax归一化）"""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # softmax确保权重和为1，代表样本分配给各专家的概率
        return nn.functional.softmax(self.linear(x), dim=-1)

# ---------------------- 2. 定义MOE模型（含负载均衡损失） ----------------------
class MoEWithLoadBalance(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        # 初始化多个专家
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        # 初始化门控网络
        self.gating = GatingNetwork(input_dim, num_experts)
    
    def forward(self, x):
        """
        前向传播：
        1. 门控网络输出权重 -> 分配样本到专家
        2. 各专家计算输出 -> 加权求和得到最终输出
        3. 计算负载均衡辅助损失（关键！）
        """
        batch_size = x.shape[0]
        
        # 步骤1：门控网络输出权重 (batch_size, num_experts)
        gate_weights = self.gating(x)
        
        # 步骤2：计算所有专家的输出 (num_experts, batch_size, output_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        # 步骤3：加权求和得到最终输出 (batch_size, output_dim)
        # 先调整维度：gate_weights (batch_size, num_experts, 1)
        gate_weights_expanded = gate_weights.unsqueeze(-1)
        # 加权求和：sum(专家输出 * 对应权重)
        final_output = torch.sum(expert_outputs.permute(1, 0, 2) * gate_weights_expanded, dim=1)
        
        # 步骤4：计算负载均衡辅助损失（核心！）
        # 思路：计算每个专家的平均负载（所有样本分配给该专家的权重之和 / 总样本数）
        expert_loads = torch.sum(gate_weights, dim=0) / batch_size  # (num_experts,)
        # 用「负载的方差」作为惩罚项：方差越大，说明负载越不均，损失越大
        load_balance_loss = torch.var(expert_loads)
        
        return final_output, load_balance_loss

# ---------------------- 3. 训练流程（验证负载均衡逻辑） ----------------------
def train_moe():
    # 超参数
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    num_experts = 4  # 4个专家
    batch_size = 32
    epochs = 100
    lr = 1e-3
    lambda_balance = 0.1  # 负载均衡损失的权重（可调）
    
    # 初始化模型、损失函数（主任务：分类）、优化器
    model = MoEWithLoadBalance(input_dim, hidden_dim, output_dim, num_experts)
    criterion_main = nn.CrossEntropyLoss()  # 主任务损失（分类）
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 模拟数据：输入+标签
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播：得到主输出 + 负载均衡损失
        output, load_loss = model(x)
        
        # 主损失：分类损失
        main_loss = criterion_main(output, y)
        
        # 总损失 = 主损失 + 权重*负载均衡损失（核心！联合优化）
        total_loss = main_loss + lambda_balance * load_loss
        
        # 反向传播 + 优化
        total_loss.backward()
        optimizer.step()
        
        # 打印日志：观察负载均衡损失和专家负载
        if (epoch + 1) % 10 == 0:
            # 计算当前各专家的负载
            gate_weights = model.gating(x)
            expert_loads = torch.sum(gate_weights, dim=0) / batch_size
            print(f"Epoch {epoch+1}:")
            print(f"  主损失: {main_loss.item():.4f}, 负载均衡损失: {load_loss.item():.4f}")
            print(f"  各专家负载: {[f'{load:.4f}' for load in expert_loads.tolist()]}")
            print("-" * 50)

# 运行训练
if __name__ == "__main__":
    train_moe()