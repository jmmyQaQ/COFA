import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# 尝试导入 dgl，如果没有安装则跳过 (会影响加载 DGL 格式数据)
try:
    import dgl
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    print("Warning: DGL not installed. Cannot load DGL-format binary files.")

# === 工具函数 ===
def robust_load_edges(edge_path):
    try:
        edges = np.genfromtxt(edge_path, dtype=float)
    except:
        df = pd.read_csv(edge_path, sep='\s+', header=None, comment='#')
        edges = df.values
        
    if edges.ndim == 1:
        if edges.shape[0] == 0: raise ValueError(f"Edge file {edge_path} is empty.")
    
    mask_valid = ~np.isnan(edges).any(axis=1)
    edges = edges[mask_valid]
    edges = edges.astype(int)

    mask_pos = (edges >= 0).all(axis=1)
    edges = edges[mask_pos]

    if edges.size > 0 and edges.min() == 1:
        edges = edges - 1
        
    return edges

def feature_norm(features):
    min_values = features.min(0)
    max_values = features.max(0)
    return 2 * (features - min_values) / (max_values - min_values) - 1

def dgl_to_pyg(g):
    """
    将 DGLGraph 转换为 PyG Data 对象
    """
    # 1. 提取边 (DGL -> PyG edge_index)
    u, v = g.edges()
    edge_index = torch.stack([u, v], dim=0).long()
    
    # 2. 提取节点特征 (ndata)
    # 尝试常见的键名
    x = g.ndata.get('feat', g.ndata.get('features', g.ndata.get('x')))
    y = g.ndata.get('label', g.ndata.get('labels', g.ndata.get('y')))
    sens = g.ndata.get('sens', g.ndata.get('sensitive', g.ndata.get('val'))) # NIFA 有时用 val 代表 sensitive
    
    # 如果找不到，尝试查找含有 'sens' 字样的 key
    if sens is None:
        for key in g.ndata.keys():
            if 'sens' in key:
                sens = g.ndata[key]
                break
    
    if x is None:
        raise ValueError("DGL Graph missing 'feat' or 'x' in ndata.")
    
    # 3. 构建 Data
    data = Data(x=x, edge_index=edge_index, y=y)
    if sens is not None:
        data.sens = sens.float() # 转换为 float 供攻击模型使用
    
    return data

# === 核心加载逻辑 (PyG + DGL + CSV) ===
def load_fairness_dataset(dataset, root_dir='./data'):
    dataset = dataset.lower()
    
    configs = {
        'german': {'dir': 'german', 'csv': 'german.csv', 'edge': 'german_edges.txt', 'sens': 'Gender', 'label': 'GoodCustomer'},
        'bail':   {'dir': 'bail', 'csv': 'bail.csv', 'edge': 'bail_edges.txt', 'sens': 'WHITE', 'label': 'RECID'},
        'credit': {'dir': 'credit', 'csv': 'credit.csv', 'edge': 'credit_edges.txt', 'sens': 'Age', 'label': 'NoDefaultNextMonth'},
        'pokec_z': {'dir': 'pokec_z', 'csv': 'region_job.csv', 'edge': 'region_job_relationship.txt', 'sens': 'region', 'label': 'I_am_working_in_field'},
        'pokec_n': {'dir': 'pokec_n', 'csv': 'region_job_2.csv', 'edge': 'region_job_2_relationship.txt', 'sens': 'region', 'label': 'I_am_working_in_field'},
        'dblp':   {'dir': 'dblp', 'csv': 'dblp.csv', 'edge': 'dblp.txt', 'sens': 'None', 'label': 'None'}
    }

    if dataset not in configs:
        raise NotImplementedError(f"Dataset {dataset} not supported.")

    cfg = configs[dataset]
    base_path = os.path.join(root_dir, cfg['dir'])
    
    # ---------------------------------------------------------
    # 1. 优先尝试加载二进制文件 (支持 PyTorch, NumPy, DGL)
    # ---------------------------------------------------------
    binary_candidates = [
        f"{dataset}.bin", f"{dataset}.pt",
        f"{dataset.upper()}.bin", f"{dataset.upper()}.pt",
        "dblp.bin", "dblp.pt"
    ]
    
    pt_path = None
    for name in binary_candidates:
        p = os.path.join(base_path, name)
        if os.path.exists(p):
            pt_path = p
            break
    
    if pt_path is not None:
        print(f"[{dataset}] Found binary file at {pt_path}")
        data = None
        
        # --- 策略 A: DGL Load (针对 NIFA 数据集) ---
        if HAS_DGL:
            try:
                # 不打印 DGL 的后端信息
                glist, _ = dgl.load_graphs(pt_path)
                print(f"  > Loaded successfully with dgl.load_graphs")
                g = glist[0]
                data = dgl_to_pyg(g)
                print(f"  > Converted DGL graph to PyG Data object.")
            except Exception as e_dgl:
                # 不是 DGL 格式，或者加载失败，继续尝试下一个
                pass
        
        # --- 策略 B: PyTorch Load ---
        if data is None:
            try:
                data_obj = torch.load(pt_path)
                if isinstance(data_obj, dict):
                    # 字典转 Data
                    x = data_obj.get('x', data_obj.get('features'))
                    y = data_obj.get('y', data_obj.get('labels'))
                    sens = data_obj.get('sens', data_obj.get('sensitive'))
                    edge_index = data_obj.get('edge_index', data_obj.get('adj'))
                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.sens = sens
                elif isinstance(data_obj, Data):
                    data = data_obj
                print(f"  > Loaded successfully with torch.load")
            except:
                pass

        # --- 策略 C: NumPy Load ---
        if data is None:
            try:
                data_obj = np.load(pt_path, allow_pickle=True)
                if isinstance(data_obj, np.ndarray) and data_obj.ndim == 0:
                    data_obj = data_obj.item()
                
                # 同样的字典处理逻辑
                if isinstance(data_obj, dict):
                    x = torch.tensor(data_obj.get('x'))
                    y = torch.tensor(data_obj.get('y'))
                    sens = torch.tensor(data_obj.get('sens'))
                    edge_index = torch.tensor(data_obj.get('edge_index'))
                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.sens = sens
                print(f"  > Loaded successfully with numpy.load")
            except:
                pass
        
        # --- 最终检查 ---
        if data is not None:
            # 属性检查
            if not hasattr(data, 'sens') or data.sens is None:
                 print(f"Warning: Loaded data missing 'sens'. Keys: {data.keys if hasattr(data, 'keys') else 'unknown'}")
                 raise ValueError(f"Loaded {dataset} binary data missing 'sens' attribute.")
            
            data.sens = data.sens.float()
            if not hasattr(data, 'num_nodes') or data.num_nodes is None:
                data.num_nodes = data.x.shape[0]
            
            print(f"  > Done! Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
            return data
        else:
            print(f"  > Failed to load binary file with DGL, Torch, or Numpy.")

    # ---------------------------------------------------------
    # 2. 回退到 CSV 加载模式
    # ---------------------------------------------------------
    print(f"[{dataset}] Binary file load failed or not found, trying CSV loading...")
    
    csv_path = os.path.join(base_path, cfg['csv'])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Neither binary nor CSV file found for {dataset} at {base_path}")
        
    df = pd.read_csv(csv_path)
    
    header = list(df.columns)
    if cfg['label'] in header: header.remove(cfg['label'])
    
    if dataset == 'german':
        if 'OtherLoansAtStore' in header: header.remove('OtherLoansAtStore')
        if 'PurposeOfLoan' in header: header.remove('PurposeOfLoan')
        df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    elif dataset == 'credit':
        if 'Single' in header: header.remove('Single')
    elif dataset.startswith('pokec'):
        if df[cfg['sens']].dtype == object:
             top_val = df[cfg['sens']].mode()[0]
             df[cfg['sens']] = (df[cfg['sens']] == top_val).astype(int)
        else:
             df[cfg['sens']] = (df[cfg['sens']] > 0).astype(int)

    labels = df[cfg['label']].values
    labels[labels == -1] = 0
    labels = torch.LongTensor(labels)
    
    sens = df[cfg['sens']].values.astype(int)
    sens = torch.FloatTensor(sens) 
    
    feat_data = df[header]
    features = pd.get_dummies(feat_data).values.astype(np.float32)
    features = feature_norm(features)
    features = torch.FloatTensor(features)
    
    edge_path = os.path.join(base_path, cfg['edge'])
    edges = robust_load_edges(edge_path)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    if dataset not in ['pokec_z', 'pokec_n']: 
        row, col = edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)

    data = Data(x=features, edge_index=edge_index, y=labels)
    data.sens = sens
    data.num_nodes = features.shape[0]
    
    return data