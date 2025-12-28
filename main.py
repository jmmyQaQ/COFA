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
    
    print(f"\n" + "="*60)
    print(f"   COFA: Co-Optimized Fairness Attack under LDP (Fixed)")
    print(f"="*60)
    print(f"Target Dataset:  {args.dataset}")
    print(f"Surrogate Model: {args.surrogate_model}")
    print(f"LDP Noise Rate:  {args.noise_rate}")
    print(f"Lambda Fair:     {args.lambda_fair}")
    print(f"Attack Budget:   {args.ptb_rate}")
    print(f"="*60 + "\n")

    # 1. 加载数据
    data = load_fairness_dataset(args.dataset, args.data_path).to(device)
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    
    # LDP 加噪
    S_clean = data.sens
    S_noisy = apply_ldp_noise(S_clean, args.noise_rate).to(device)

    # 2. 模型初始化
    # Estimator: 负责穿透 LDP 噪声
    estimator = SensitiveEstimator(data.num_features, args.hidden_dim).to(device)
    opt_est = optim.Adam(estimator.parameters(), lr=args.lr_est)

    # Surrogate: 代理受害者
    surrogate = SurrogateModel(args.surrogate_model, data.num_features, args.hidden_dim, int(data.y.max().item())+1).to(device)
    opt_sur = optim.Adam(surrogate.parameters(), lr=args.lr_sur)

    # 3. 攻击结构初始化
    num_candidates = min(num_edges, 200000) 
    pos_edge_index = data.edge_index
    
    # 随机采样非边
    neg_row = torch.randint(0, num_nodes, (num_candidates,), device=device)
    neg_col = torch.randint(0, num_nodes, (num_candidates,), device=device)
    neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
    
    candidate_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    
    base_w = torch.cat([
        torch.ones(pos_edge_index.shape[1], device=device),
        torch.zeros(neg_edge_index.shape[1], device=device)
    ])
    
    # P: 扰动参数
    P = torch.zeros(candidate_edge_index.shape[1], device=device, requires_grad=True)
    opt_atk = optim.Adam([P], lr=args.lr_atk)

    print(f"[COFA] Initialized candidates: {candidate_edge_index.shape[1]} edges")

    # 为了训练稳定，建议固定 Train Mask，而不是每个 Epoch 随机
    # 这里模拟 50% 训练数据用于 Surrogate 训练
    train_mask = torch.rand(num_nodes, device=device) < 0.5 

    # --- COFA 联合训练循环 ---
    for epoch in range(args.epochs):
        estimator.train()
        surrogate.train()
        
        # A. 生成可微图结构
        edge_weight_soft = torch.sigmoid(base_w + P)
        
        # B. Estimator 前向传播
        # Estimator 利用当前的毒药图来预测敏感属性
        s_logits = estimator(data.x, candidate_edge_index, edge_weight_soft)
        s_probs = torch.sigmoid(s_logits).squeeze()
        
        # Loss 1: Estimation Loss (使用修正后的 Backward Correction)
        # 这一步会更新 Estimator (使其更准) 和 P (使图结构包含更多敏感信息 -> 增加同质性)
        loss_est = backward_correction_loss(s_logits.squeeze(), S_noisy, args.noise_rate)
        
        # C. Surrogate 前向传播
        y_logits = surrogate(data.x, candidate_edge_index, edge_weight_soft)
        
        # Loss 2: Utility Loss (保持图的可用性)
        loss_util = F.cross_entropy(y_logits[train_mask], data.y[train_mask])
        
        # D. Loss 3: Fairness Attack Loss
        # 预测的正类概率
        y_probs = F.softmax(y_logits, dim=1)[:, 1] 
        
        # 【关键修改】使用 detach() 截断梯度
        # 我们希望 P 改变图结构从而改变 y_probs，进而拉大 Gap
        # 但我们不希望 Estimator 为了拉大 Gap 而故意预测错误
        s_probs_detached = s_probs.detach()
        
        # (可选优化) 基于熵的加权：只信任预测置信度高的节点
        # entropy = - (s_probs_detached * torch.log(s_probs_detached+1e-6) + (1-s_probs_detached) * torch.log(1-s_probs_detached+1e-6))
        # weight = 1 - entropy
        
        prob_s1 = s_probs_detached
        prob_s0 = 1 - s_probs_detached
        
        # 计算加权平均预测值
        mean_s1 = (y_probs * prob_s1).sum() / (prob_s1.sum() + 1e-6)
        mean_s0 = (y_probs * prob_s0).sum() / (prob_s0.sum() + 1e-6)
        
        fairness_gap = torch.abs(mean_s1 - mean_s0)
        
        # 攻击目标：最大化 Gap
        loss_attack = - fairness_gap
        
        # E. 联合反向传播
        # 注意：
        # - loss_est: 更新 Estimator, P
        # - loss_attack: 更新 Surrogate, P (因为 s_probs 没梯度，所以只通过 y_probs 回传给 P)
        # - loss_util: 更新 Surrogate, P
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
                est_acc, est_auc = evaluate_performance(s_logits, S_clean)
                
                # 计算真实 Gap (基于干净标签，攻击者不可见，仅用于评估)
                real_gap = torch.abs(
                    y_probs[S_clean==1].mean() - y_probs[S_clean==0].mean()
                ).item()
                
            print(f"Epoch {epoch+1:03d} | Total: {total_loss.item():.2f} | "
                  f"Est Acc: {est_acc:.4f} (AUC: {est_auc:.4f}) | "
                  f"Est Loss: {loss_est.item():.4f} | "
                  f"Gap (Est): {fairness_gap.item():.4f} / (Real): {real_gap:.4f}")

    # --- 导出最终毒药图 ---
    print("\n[COFA] Generating Discrete Poisoned Graph...")
    with torch.no_grad():
        final_scores = base_w + P
        current_num_edges = pos_edge_index.shape[1]
        budget_edges = int(current_num_edges * (1 + args.ptb_rate)) 
        
        _, top_indices = torch.topk(final_scores, budget_edges)
        final_edge_index = candidate_edge_index[:, top_indices]
        
        save_dir = f'./data/{args.dataset}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = f'{save_dir}/COFA_poisoned_{args.surrogate_model}.pt'
        torch.save(final_edge_index, save_path)
        
        print(f"Saved poisoned adjacency to: {save_path}")
        print(f"Original Edges: {current_num_edges}")
        print(f"Poisoned Edges: {final_edge_index.shape[1]}")

if __name__ == "__main__":
    train_cofa()