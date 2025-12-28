import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
from dataset import load_fairness_dataset
from config import get_args
from model import SensitiveEstimator, SurrogateModel
from utils import backward_correction_loss, apply_ldp_noise, evaluate_performance

def set_seed(seed):
    """固定训练过程中的随机种子，确保生成的毒药图可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_cofa():
    args = get_args()
    device = torch.device(args.device)
    
    # 【关键修改】固定训练种子
    set_seed(args.seed)
    
    print(f"\n" + "="*60)
    print(f"   COFA: Co-Optimized Fairness Attack under LDP (Final Optimized)")
    print(f"="*60)
    print(f"Target Dataset:  {args.dataset}")
    print(f"Surrogate Model: {args.surrogate_model}")
    print(f"LDP Noise Rate:  {args.noise_rate}")
    print(f"Lambda Fair:     {args.lambda_fair}")
    print(f"Attack Budget:   {args.ptb_rate}")
    print(f"Random Seed:     {args.seed}")
    print(f"="*60 + "\n")

    # 1. 加载数据
    data = load_fairness_dataset(args.dataset, args.data_path).to(device)
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    
    # LDP 加噪
    S_clean = data.sens
    S_noisy = apply_ldp_noise(S_clean, args.noise_rate).to(device)

    # 2. 模型初始化
    estimator = SensitiveEstimator(data.num_features, args.hidden_dim).to(device)
    opt_est = optim.Adam(estimator.parameters(), lr=args.lr_est)

    surrogate = SurrogateModel(args.surrogate_model, data.num_features, args.hidden_dim, int(data.y.max().item())+1).to(device)
    opt_sur = optim.Adam(surrogate.parameters(), lr=args.lr_sur)

    # 3. 攻击结构初始化
    pos_edge_index = data.edge_index
    
    # 动态调整负采样数量
    min_neg_needed = int(num_edges * args.ptb_rate * 1.5)
    num_candidates_limit = 2000000 
    num_neg_samples = min(num_nodes * num_nodes, max(200000, min_neg_needed))
    num_neg_samples = min(num_neg_samples, num_candidates_limit)

    print(f"[Init] Positive Edges: {num_edges}, Planned Negative Samples: {num_neg_samples}")
    
    neg_row = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    neg_col = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
    
    candidate_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    
    base_w = torch.cat([
        torch.ones(pos_edge_index.shape[1], device=device),
        torch.zeros(neg_edge_index.shape[1], device=device)
    ])
    
    P = torch.zeros(candidate_edge_index.shape[1], device=device, requires_grad=True)
    opt_atk = optim.Adam([P], lr=args.lr_atk)

    print(f"[COFA] Total candidates pool: {candidate_edge_index.shape[1]} edges")

    train_mask = torch.rand(num_nodes, device=device) < 0.5 

    # --- COFA 联合训练循环 ---
    for epoch in range(args.epochs):
        estimator.train()
        surrogate.train()
        
        # A. 生成可微图结构 (Gumbel-Softmax)
        logits = base_w + P
        temp = 1.0 
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        edge_weight_soft = torch.sigmoid((logits + gumbel_noise) / temp)
        
        # B. Estimator 前向传播
        s_logits = estimator(data.x, candidate_edge_index, edge_weight_soft)
        s_probs = torch.sigmoid(s_logits).squeeze()
        loss_est = backward_correction_loss(s_logits.squeeze(), S_noisy, args.noise_rate)
        
        # C. Surrogate 前向传播
        y_logits = surrogate(data.x, candidate_edge_index, edge_weight_soft)
        loss_util = F.cross_entropy(y_logits[train_mask], data.y[train_mask])
        
        # D. Loss 3: Fairness Attack Loss
        y_probs = F.softmax(y_logits, dim=1)[:, 1] 
        s_probs_detached = s_probs.detach()
        prob_s1 = s_probs_detached
        prob_s0 = 1 - s_probs_detached
        
        mean_s1 = (y_probs * prob_s1).sum() / (prob_s1.sum() + 1e-6)
        mean_s0 = (y_probs * prob_s0).sum() / (prob_s0.sum() + 1e-6)
        
        fairness_gap = torch.abs(mean_s1 - mean_s0)
        loss_attack = - fairness_gap
        
        total_loss = loss_est + args.lambda_fair * loss_attack + 0.5 * loss_util
        
        opt_est.zero_grad()
        opt_sur.zero_grad()
        opt_atk.zero_grad()
        total_loss.backward()
        opt_est.step()
        opt_sur.step()
        opt_atk.step()
        
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                est_acc, est_auc = evaluate_performance(s_logits, S_clean)
                real_gap = torch.abs(y_probs[S_clean==1].mean() - y_probs[S_clean==0].mean()).item()
            print(f"Epoch {epoch+1:03d} | Total: {total_loss.item():.2f} | "
                  f"Est Acc: {est_acc:.4f} (AUC: {est_auc:.4f}) | "
                  f"Gap (Est): {fairness_gap.item():.4f} / (Real): {real_gap:.4f}")

    # --- 导出最终毒药图 ---
    print("\n[COFA] Generating Discrete Poisoned Graph...")
    with torch.no_grad():
        final_scores = base_w + P
        
        # 权重分布调试
        probs = torch.sigmoid(final_scores)
        print("-" * 30)
        print("[DEBUG] Weight Distribution Analysis:")
        print(f"  Count > 0.9: {(probs > 0.9).sum().item()}")
        print(f"  Count < 0.1: {(probs < 0.1).sum().item()}")
        print("-" * 30)

        # 安全 Budget 计算
        current_num_edges = pos_edge_index.shape[1]
        target_budget = int(current_num_edges * (1 + args.ptb_rate)) 
        budget_edges = min(target_budget, final_scores.shape[0])
        
        _, top_indices = torch.topk(final_scores, budget_edges)
        final_edge_index = candidate_edge_index[:, top_indices]
        
        save_dir = f'./data/{args.dataset}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 文件名加入 seed 标记，防止混淆
        save_path = f'{save_dir}/COFA_poisoned_{args.surrogate_model}_seed{args.seed}.pt'
        torch.save(final_edge_index, save_path)
        
        print(f"Saved poisoned adjacency to: {save_path}")

if __name__ == "__main__":
    train_cofa()