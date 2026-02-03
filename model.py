import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, APPNP

# === 新增: GRASP 扰动层 ===
def grasp_perturbation(x, sigma, p, training=True):
    """
    GRASP: Structured Perturbation
    Paper: Differentially Private Graph Reconstruction Defense with Structured Perturbation
    """
    if not training or sigma <= 0:
        return x
        
    device = x.device
    num_nodes, dim = x.shape
    
    # 1. 生成噪声向量 [cite: 339]
    # 独立噪声 (Independent Noise)
    noise_ind = torch.randn_like(x, device=device) * sigma
    # 相同噪声 (Identical Noise) - 所有节点共享同一个噪声向量
    noise_identical = torch.randn(1, dim, device=device) * sigma
    
    # 2. 伯努利采样混合 [cite: 344]
    # lambda_i ~ Bernoulli(p)
    # p 是使用 identical noise 的概率
    bernoulli_mask = torch.bernoulli(torch.full((num_nodes, 1), p, device=device))
    
    # 3. 组合噪声
    final_noise = bernoulli_mask * noise_identical + (1 - bernoulli_mask) * noise_ind
    
    # 4. 加噪并归一化 [cite: 345]
    x_perturbed = x + final_noise
    
    # 论文公式 (12) 建议使用 LayerNorm 或 L2-Norm
    # 这里使用 L2 归一化以保持数值稳定
    x_perturbed = F.normalize(x_perturbed, p=2, dim=1)
    
    return x_perturbed

# --- 敏感属性估计器 ---
class SensitiveEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(SensitiveEstimator, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1) 
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# --- 代理受害者模型 (支持 GRASP) ---
class SurrogateModel(nn.Module):
    def __init__(self, model_name, input_dim, hidden_dim, output_dim, dropout=0.5, 
                 enable_grasp=False, grasp_sigma=0.0, grasp_p=0.5):
        super(SurrogateModel, self).__init__()
        self.model_name = model_name
        self.dropout = dropout
        
        # GRASP 参数
        self.enable_grasp = enable_grasp
        self.grasp_sigma = grasp_sigma
        self.grasp_p = grasp_p
        
        if model_name == 'GCN' or model_name == 'FairGNN': 
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
            
        elif model_name == 'GraphSAGE':
            self.conv1 = GraphConv(input_dim, hidden_dim, aggr='mean')
            self.conv2 = GraphConv(hidden_dim, output_dim, aggr='mean')
            
        elif model_name == 'APPNP':
            self.lin1 = nn.Linear(input_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, output_dim)
            self.prop = APPNP(K=10, alpha=0.1)
        else:
            raise ValueError(f"Model {model_name} not supported")

    def forward(self, x, edge_index, edge_weight=None):
        if self.model_name == 'APPNP':
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.lin1(x))
            
            # --- Apply GRASP if enabled (on hidden rep) ---
            if self.enable_grasp:
                x = grasp_perturbation(x, self.grasp_sigma, self.grasp_p, self.training)
                
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            x = self.prop(x, edge_index, edge_weight)
            return x
            
        else: # GCN, GraphSAGE
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            
            # --- Apply GRASP if enabled (Paper: on aggregated embeddings) ---
            if self.enable_grasp:
                x = grasp_perturbation(x, self.grasp_sigma, self.grasp_p, self.training)
            
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x