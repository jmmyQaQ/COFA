import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, APPNP

# --- 敏感属性估计器 (始终使用 GCN 即可) ---
class SensitiveEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(SensitiveEstimator, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1) # 输出 1维 Logits
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# --- 代理受害者模型 (支持多种架构) ---
class SurrogateModel(nn.Module):
    def __init__(self, model_name, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(SurrogateModel, self).__init__()
        self.model_name = model_name
        self.dropout = dropout
        
        if model_name == 'GCN' or model_name == 'FairGNN': 
            # FairGNN 的 backbone 也是 GCN
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
            
        elif model_name == 'GraphSAGE':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
            
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
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            # APPNP 传播支持 edge_weight
            x = self.prop(x, edge_index, edge_weight)
            return x
            
        else: # GCN, GraphSAGE, FairGNN(Backbone)
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x