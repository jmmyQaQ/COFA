import os
import torch
import numpy as np
import pandas as pd
import dgl
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# === 移植自 NIFA 的工具函数 ===
def robust_load_edges(edge_path):
    """
    健壮的边加载函数：处理表头、负数、NaN、1-based索引
    """
    try:
        edges = np.genfromtxt(edge_path, dtype=float)
    except:
        df = pd.read_csv(edge_path, sep='\s+', header=None, comment='#')
        edges = df.values
        
    if edges.ndim == 1:
        if edges.shape[0] == 0: raise ValueError(f"Edge file {edge_path} is empty.")
        pass

    # 过滤无效行
    mask_valid = ~np.isnan(edges).any(axis=1)
    edges = edges[mask_valid]
    edges = edges.astype(int)

    # 过滤负数
    mask_pos = (edges >= 0).all(axis=1)
    edges = edges[mask_pos]

    # 1-based -> 0-based
    if edges.size > 0 and edges.min() == 1:
        print("  > Detected 1-based indexing, converting to 0-based.")
        edges = edges - 1
        
    return edges

def feature_norm(features):
    min_values = features.min(0)
    max_values = features.max(0)
    return 2 * (features - min_values) / (max_values - min_values) - 1

# === 主要加载逻辑 ===
def load_fairness_dataset(dataset, root_dir='./data'):
    """
    加载数据并返回 PyG Data 对象
    """
    dataset = dataset.lower()
    
    # 配置字典 (与 NIFA 保持一致)
    configs = {
        'german': {'dir': 'german', 'csv': 'german.csv', 'edge': 'german_edges.txt', 'sens': 'Gender', 'label': 'GoodCustomer'},
        'bail':   {'dir': 'bail', 'csv': 'bail.csv', 'edge': 'bail_edges.txt', 'sens': 'WHITE', 'label': 'RECID'},
        'credit': {'dir': 'credit', 'csv': 'credit.csv', 'edge': 'credit_edges.txt', 'sens': 'Age', 'label': 'NoDefaultNextMonth'},
        'pokec_z': {'dir': 'pokec', 'csv': 'region_job.csv', 'edge': 'region_job_relationship.txt', 'sens': 'region', 'label': 'I_am_working_in_field'},
        'pokec_n': {'dir': 'pokec', 'csv': 'region_job_2.csv', 'edge': 'region_job_2_relationship.txt', 'sens': 'region', 'label': 'I_am_working_in_field'},
        'dblp':   {'dir': 'dblp', 'csv': 'dblp.csv', 'edge': 'dblp.txt', 'sens': 'None', 'label': 'None'} # DBLP处理较特殊需自行适配列名
    }

    if dataset not in configs:
        raise NotImplementedError(f"Dataset {dataset} not supported.")

    cfg = configs[dataset]
    base_path = os.path.join(root_dir, cfg['dir'])
    
    # 1. 读取 CSV 特征与标签
    csv_path = os.path.join(base_path, cfg['csv'])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 特殊预处理 (NIFA Logic)
    header = list(df.columns)
    if cfg['label'] in header: header.remove(cfg['label'])
    
    if dataset == 'german':
        if 'OtherLoansAtStore' in header: header.remove('OtherLoansAtStore')
        if 'PurposeOfLoan' in header: header.remove('PurposeOfLoan')
        df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    elif dataset == 'credit':
        if 'Single' in header: header.remove('Single')
    elif dataset.startswith('pokec'):
        # Pokec 特殊处理: region 通常作为敏感属性，需要二值化
        if df[cfg['sens']].dtype == object:
             top_val = df[cfg['sens']].mode()[0]
             df[cfg['sens']] = (df[cfg['sens']] == top_val).astype(int)
        else:
             df[cfg['sens']] = (df[cfg['sens']] > 0).astype(int)

    # 提取 Label, Sensitive, Features
    labels = df[cfg['label']].values
    labels[labels == -1] = 0
    labels = torch.LongTensor(labels)
    
    sens = df[cfg['sens']].values.astype(int)
    sens = torch.FloatTensor(sens) # E2E中需要浮点数计算Loss
    
    # 特征处理
    feat_data = df[header]
    # 简单的一热编码处理非数值列
    features = pd.get_dummies(feat_data).values.astype(np.float32)
    features = feature_norm(features) # 归一化
    features = torch.FloatTensor(features)
    
    # 2. 读取 Edges
    edge_path = os.path.join(base_path, cfg['edge'])
    edges = robust_load_edges(edge_path)
    
    # 转换为 PyG edge_index [2, E]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # 转为无向图
    if dataset not in ['pokec_z', 'pokec_n']: # Pokec 通常本身就是很大的有向/无向混合，PyG处理需要注意
        # 简单起见，强制无向
        row, col = edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)

    # 3. 封装 Data 对象
    data = Data(x=features, edge_index=edge_index, y=labels)
    data.sens = sens
    data.num_nodes = features.shape[0]
    
    return data