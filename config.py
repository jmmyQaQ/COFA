import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # --- 基础环境 ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    
    # --- 数据集 ---
    # 支持 NIFA 框架中的所有数据集
    parser.add_argument('--dataset', type=str, default='german', 
                        choices=['pokec_z', 'pokec_n', 'dblp', 'german', 'bail', 'credit'])
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset folder.')
    
    # --- LDP 设置 ---
    parser.add_argument('--noise_rate', type=float, default=0.1, help='LDP flip rate (rho).')
    
    # --- 模型选择 ---
    parser.add_argument('--surrogate_model', type=str, default='GCN', 
                        choices=['GCN', 'GraphSAGE', 'APPNP', 'FairGNN'],
                        help='Backbone model for the surrogate victim.')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # --- 端到端 (E2E) 攻击参数 ---
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr_est', type=float, default=0.01, help='LR for Estimator.')
    parser.add_argument('--lr_sur', type=float, default=0.01, help='LR for Surrogate.')
    parser.add_argument('--lr_atk', type=float, default=0.1, help='LR for P (Attack Structure).')
    
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='Budget: ratio of edges to modify.')
    
    # 核心：联合优化权重
    # Total = Loss_Est + lambda * Loss_Fair_Attack
    parser.add_argument('--lambda_fair', type=float, default=1.0, help='Weight for maximizing unfairness.')
    
    args = parser.parse_args()
    return args