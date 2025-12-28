import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from dataset import load_fairness_dataset
from config import get_args
from model import SensitiveEstimator, SurrogateModel
from utils import backward_correction_loss, apply_ldp_noise, evaluate_performance

def train_cofa():
    args = get_args()
    device = torch.device(args.device)
    
    # 打印 COFA 启动横幅
    print(f"\n" + "="*60)
    print(f"   COFA: Co-Optimized Fairness Attack under LDP")
    print(f"="*60)
    print(f"Target Dataset:  {args.dataset}")
    print(f"Surrogate Model: {args.surrogate_model}")
    print(f"LDP Noise Rate:  {args.noise_rate}")
    print(f"Lambda Fair:     {args.lambda_fair}")
    print(f"Attack Budget:   {args.ptb_rate}")
    print(f"="*60 + "\n")

    # 1. 加载数据 (PyG Data Format)
    data = load_fairness_dataset(args.dataset, args.data_path).to(device)
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    
    # LDP 加噪 (模拟攻击者视角)
    S_clean = data.sens
    S_noisy = apply_ldp_noise(S_clean, args.noise_rate).to(device)

    # 2. 模型初始化
    # Estimator: 负责穿透 LDP 噪声，恢复敏感属性分布
    estimator = SensitiveEstimator(data.num_features, args.hidden_dim).to(device)
    opt_est = optim.Adam(estimator.parameters(), lr=args.lr_est)

    # Surrogate: 代理受害者，用于计算 Fairness Gap 的梯度回传
    surrogate = SurrogateModel(args.surrogate_model, data.num_features, args.hidden_dim, int(data.y.max().item())+1).to(device)
    opt_sur = optim.Adam(surrogate.parameters(), lr=args.lr_sur)

    # 3. 攻击结构初始化 (Sparse Candidates to avoid OOM)
    # 策略：只优化 "现有边" + "随机采样的非边"
    num_candidates = min(num_edges, 200000) 
    
    pos_edge_index = data.edge_index
    
    # 随机采样非边 (Negative Edges)
    neg_row = torch.randint(0, num_nodes, (num_candidates,), device=device)
    neg_col = torch.randint(0, num_nodes, (num_candidates,), device=device)
    neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
    
    # 合并为候选集
    candidate_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    
    # 基础权重: 现有边=1, 非边=0
    base_w = torch.cat([
        torch.ones(pos_edge_index.shape[1], device=device),
        torch.zeros(neg_edge_index.shape[1], device=device)
    ])
    
    # P: COFA 的核心可学习参数 (Perturbation Matrix 的稀疏表示)
    P = torch.zeros(candidate_edge_index.shape[1], device=device, requires_grad=True)
    opt_atk = optim.Adam([P], lr=args.lr_atk)

    print(f"[COFA] Initialized candidates: {candidate_edge_index.shape[1]} edges (Pos: {pos_edge_index.shape[1]})")

    # --- COFA 联合训练循环 ---
    for epoch in range(args.epochs):
        estimator.train()
        surrogate.train()
        
        # A. 生成可微图结构 (Differentiable Graph Sampling)
        # edge_weight \in (0, 1)，允许梯度流过
        edge_weight_soft = torch.sigmoid(base_w + P)
        
        # B. Estimator 前向传播
        # 利用当前的图结构预测敏感属性
        s_logits = estimator(data.x, candidate_edge_index, edge_weight_soft)
        s_probs = torch.sigmoid(s_logits).squeeze()
        
        # Loss 1: Estimation Loss (使用 Backward Correction 对抗 LDP)
        loss_est = backward_correction_loss(s_logits.squeeze(), S_noisy, args.noise_rate)
        
        # C. Surrogate 前向传播
        # 在当前的毒药图上训练下游任务
        y_logits = surrogate(data.x, candidate_edge_index, edge_weight_soft)
        
        # Loss 2: Utility Loss (保持图的可用性，防止被检测)
        # 使用随机 mask 模拟训练集划分
        train_mask = torch.rand(num_nodes, device=device) < 0.5 
        loss_util = F.cross_entropy(y_logits[train_mask], data.y[train_mask])
        
        # D. Loss 3: Fairness Attack Loss (COFA 核心)
        # 计算 Differentiable Statistical Parity (SP)
        # 关键：这里使用的是 estimator 预测的 s_probs，而不是固定的标签
        # 这样梯度会同时优化 P (让图更不公平) 和 Estimator (让分组更准)
        
        y_probs = F.softmax(y_logits, dim=1)[:, 1] # 假设 class 1 为正类
        
        # Soft Group Assignment
        prob_s1 = s_probs
        prob_s0 = 1 - s_probs
        
        # 计算两组的平均预测概率 (Soft Mean)
        mean_s1 = (y_probs * prob_s1).sum() / (prob_s1.sum() + 1e-6)
        mean_s0 = (y_probs * prob_s0).sum() / (prob_s0.sum() + 1e-6)
        
        # Fairness Gap
        fairness_gap = torch.abs(mean_s1 - mean_s0)
        
        # 攻击目标：最大化 Gap -> 最小化 -Gap
        loss_attack = - fairness_gap
        
        # E. 联合反向传播
        total_loss = loss_est + args.lambda_fair * loss_attack + 0.5 * loss_util
        
        opt_est.zero_grad()
        opt_sur.zero_grad()
        opt_atk.zero_grad()
        
        total_loss.backward()
        
        opt_est.step()
        opt_sur.step()
        opt_atk.step()
        
        # 监控日志
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                # 计算 Estimator 的真实准确率 (仅供观察)
                est_acc, _ = evaluate_performance(s_logits, S_clean)
                # 计算真实的 Gap
                real_gap = torch.abs(
                    y_probs[S_clean==1].mean() - y_probs[S_clean==0].mean()
                ).item()
                
            print(f"Epoch {epoch+1:03d} | Total: {total_loss.item():.2f} | "
                  f"Est Acc: {est_acc:.4f} | "
                  f"Attack Loss: {loss_attack.item():.4f} (Real Gap: {real_gap:.4f})")

    # --- 导出最终毒药图 (Discretization) ---
    print("\n[COFA] Generating Discrete Poisoned Graph...")
    with torch.no_grad():
        final_scores = base_w + P
        
        # 根据 Budget 截断，选取权重最大的前 K 条边
        # 允许边数量根据 ptb_rate 浮动
        current_num_edges = pos_edge_index.shape[1]
        budget_edges = int(current_num_edges * (1 + args.ptb_rate)) 
        
        _, top_indices = torch.topk(final_scores, budget_edges)
        
        final_edge_index = candidate_edge_index[:, top_indices]
        
        # 确保保存目录存在
        save_dir = f'./data/{args.dataset}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = f'{save_dir}/COFA_poisoned_{args.surrogate_model}.pt'
        torch.save(final_edge_index, save_path)
        
        print(f"Saved poisoned adjacency to: {save_path}")
        print(f"Original Edges: {current_num_edges}")
        print(f"Poisoned Edges: {final_edge_index.shape[1]}")
        print(f"[COFA] Attack Finished Successfully.")

if __name__ == "__main__":
    train_cofa()