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
    """固定训练过程中的随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def estimate_prior_entropy(s_noisy, rho):
    """
    [Information Theory Component]
    从 LDP 噪声标签中估计真实标签的先验熵 H(S)。
    用于计算互信息 I(Z; S) = H(S) - H(S|Z)。
    """
    # 1. 计算噪声数据的正类比例
    p_noisy = s_noisy.float().mean()
    
    # 2. 反解真实数据的正类比例 (Inverse LDP)
    if rho != 0.5:
        p_clean = (p_noisy - rho) / (1 - 2 * rho)
    else:
        p_clean = p_noisy 
        
    # 3. 截断防止溢出
    p_clean = torch.clamp(p_clean, 1e-6, 1 - 1e-6)
    
    # 4. 计算二元熵
    entropy = - (p_clean * torch.log(p_clean) + (1 - p_clean) * torch.log(1 - p_clean))
    return entropy

def train_cofa():
    args = get_args()
    device = torch.device(args.device)
    
    # 固定训练种子
    set_seed(args.seed)
    
    print(f"\n" + "="*60)
    print(f"   COFA: Co-Optimized Fairness Attack (Final Version)")
    print(f"   Feature: LDP + SP/EO Optimization + MI Maximization")
    print(f"="*60)
    print(f"Target Dataset:  {args.dataset}")
    print(f"Surrogate Model: {args.surrogate_model}")
    print(f"LDP Noise Rate:  {args.noise_rate}")
    print(f"Lambda Fair:     {args.lambda_fair}")
    print(f"Attack Budget:   {args.ptb_rate}")
    print(f"Random Seed:     {args.seed}")
    print(f"="*60 + "\n")

    # ------------------------------------------------------------------
    # 1. 加载数据与预处理
    # ------------------------------------------------------------------
    data = load_fairness_dataset(args.dataset, args.data_path).to(device)
    
    # 【修复 1】强制清洗敏感属性 (针对 DBLP 等多分类敏感属性崩溃问题)
    unique_sens = torch.unique(data.sens)
    if len(unique_sens) > 2 or unique_sens.max() > 1 or unique_sens.min() < 0:
        print(f"[Warning] Sensitive attributes are not binary 0/1. Unique values: {unique_sens.cpu().numpy()}")
        print(f"          > Binarizing sensitive attributes (val > 0 -> 1, else 0)...")
        data.sens = (data.sens > 0).float()
    
    # 【修复 2】计算类别权重 (针对 Pokec 等不平衡数据集的模型坍塌问题)
    y_train = data.y
    if hasattr(data, 'train_mask'):
        y_train = data.y[data.train_mask]
    
    class_counts = torch.bincount(y_train)
    class_counts = class_counts + 1 # 防止除零
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts) # 归一化
    class_weights = class_weights.to(device)
    
    print(f"[Info] Class Weights computed: {class_weights.cpu().numpy()}")

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    
    # 生成 LDP 噪声敏感属性 (这是攻击者唯一能看到的敏感信息)
    S_clean = data.sens
    S_noisy = apply_ldp_noise(S_clean, args.noise_rate).to(device)

    # ------------------------------------------------------------------
    # 2. 模型初始化 (三路优化参数)
    # ------------------------------------------------------------------
    # Group 1: 敏感属性估计器 (Estimator) - 负责最大化互信息 I(Z;S)
    estimator = SensitiveEstimator(data.num_features, args.hidden_dim).to(device)
    opt_est = optim.Adam(estimator.parameters(), lr=args.lr_est)

    # Group 2: 代理受害者模型 (Surrogate) - 负责保持 Utility
    surrogate = SurrogateModel(args.surrogate_model, data.num_features, args.hidden_dim, int(data.y.max().item())+1).to(device)
    opt_sur = optim.Adam(surrogate.parameters(), lr=args.lr_sur)

    # ------------------------------------------------------------------
    # 3. 攻击结构初始化 (攻击变量 P)
    # ------------------------------------------------------------------
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
    
    # 拼接正边和负边，形成候选边池
    candidate_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    
    # 初始化 Base Weight (原图存在的边为1，不存在的边为0)
    base_w = torch.cat([
        torch.ones(pos_edge_index.shape[1], device=device),
        torch.zeros(neg_edge_index.shape[1], device=device)
    ])
    
    # 初始化扰动变量 P (这是我们要优化的核心变量)
    P = torch.zeros(candidate_edge_index.shape[1], device=device, requires_grad=True)
    opt_atk = optim.Adam([P], lr=args.lr_atk)

    print(f"[COFA] Total candidates pool: {candidate_edge_index.shape[1]} edges")

    # 生成训练掩码 (用于计算 Utility Loss 和 EO Loss)
    if hasattr(data, 'train_mask'):
        train_mask = data.train_mask
    else:
        train_mask = torch.rand(num_nodes, device=device) < 0.5 

    # ------------------------------------------------------------------
    # 4. 联合训练循环 (Co-Optimization Loop)
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        estimator.train()
        surrogate.train()
        
        # === A. 生成可微图结构 (Gumbel-Softmax) ===
        # P -> Logits -> Soft Weights (橡皮泥图)
        logits = base_w + P
        temp = 1.0 
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        edge_weight_soft = torch.sigmoid((logits + gumbel_noise) / temp)
        
        # === B. 基于互信息最大化的敏感属性估计 ===
        # Estimator 在"软图"上猜测敏感属性
        s_logits = estimator(data.x, candidate_edge_index, edge_weight_soft)
        
        # 计算条件熵 H(S|Z) 的估计值 (Backward Correction Loss)
        h_s_given_z = backward_correction_loss(s_logits.squeeze(), S_noisy, args.noise_rate)
        
        # 计算先验熵 H(S) (仅用于展示互信息 MI 数值)
        with torch.no_grad():
            h_s = estimate_prior_entropy(S_noisy, args.noise_rate)
            
        # 计算互信息代理: I(Z; S) = H(S) - H(S|Z)
        mutual_info_proxy = h_s - h_s_given_z
        
        # 兼容变量名，用于下面的总损失计算
        loss_est = h_s_given_z
        
        # === C. Surrogate 前向传播 & Utility Loss ===
        # Surrogate 在"软图"上预测节点标签
        y_logits = surrogate(data.x, candidate_edge_index, edge_weight_soft)
        # 使用 Class Weights 防止模型坍塌
        loss_util = F.cross_entropy(y_logits[train_mask], data.y[train_mask], weight=class_weights)
        
        # === D. Fairness Attack Loss (SP + EO) ===
        # 核心逻辑：基于 Estimator 的猜测，最大化不同群体的预测差异
        
        # 1. 准备概率数据
        s_probs_detached = torch.sigmoid(s_logits).squeeze().detach()
        prob_s1 = s_probs_detached          # 估计为敏感群体 S=1 的概率
        prob_s0 = 1 - s_probs_detached      # 估计为非敏感群体 S=0 的概率
        
        # 针对多分类取第1类，或二分类取正类
        if y_logits.shape[1] == 2:
            y_pred_prob = F.softmax(y_logits, dim=1)[:, 1]
        else:
            y_pred_prob = F.softmax(y_logits, dim=1)[:, 1]

        # 2. 计算 SP Gap (Statistical Parity)
        # Formula: | E[y_hat | S=1] - E[y_hat | S=0] |
        mean_s1 = (y_pred_prob * prob_s1).sum() / (prob_s1.sum() + 1e-6)
        mean_s0 = (y_pred_prob * prob_s0).sum() / (prob_s0.sum() + 1e-6)
        loss_sp = - torch.abs(mean_s1 - mean_s0)

        # 3. 计算 EO Gap (Equalized Odds) 【新增】
        # Formula: | E[y_hat | S=1, Y=y] - E[y_hat | S=0, Y=y] |
        true_y = data.y
        
        # TPR Gap (在真实标签 Y=1 的样本中，S=1 和 S=0 的预测差异)
        mask_pos = (true_y == 1) & train_mask
        if mask_pos.sum() > 0:
            y_prob_pos = y_pred_prob[mask_pos]
            prob_s1_pos = prob_s1[mask_pos]
            prob_s0_pos = prob_s0[mask_pos]
            tpr_s1 = (y_prob_pos * prob_s1_pos).sum() / (prob_s1_pos.sum() + 1e-6)
            tpr_s0 = (y_prob_pos * prob_s0_pos).sum() / (prob_s0_pos.sum() + 1e-6)
            gap_tpr = torch.abs(tpr_s1 - tpr_s0)
        else:
            gap_tpr = torch.tensor(0.0).to(device)

        # FPR Gap (在真实标签 Y=0 的样本中，S=1 和 S=0 的预测差异)
        mask_neg = (true_y == 0) & train_mask
        if mask_neg.sum() > 0:
            y_prob_neg = y_pred_prob[mask_neg]
            prob_s1_neg = prob_s1[mask_neg]
            prob_s0_neg = prob_s0[mask_neg]
            fpr_s1 = (y_prob_neg * prob_s1_neg).sum() / (prob_s1_neg.sum() + 1e-6)
            fpr_s0 = (y_prob_neg * prob_s0_neg).sum() / (prob_s0_neg.sum() + 1e-6)
            gap_fpr = torch.abs(fpr_s1 - fpr_s0)
        else:
            gap_fpr = torch.tensor(0.0).to(device)

        loss_eo = - (gap_tpr + gap_fpr)

        # 4. 组合攻击损失 (各占一半权重)
        loss_attack = 0.5 * loss_sp + 0.5 * loss_eo
        
        # === E. 总损失与反向传播 ===
        # Co-Optimization: 同时优化 Estimator, Surrogate 和 攻击结构 P
        # 最小化 loss_est 等价于最大化 I(Z;S)
        total_loss = loss_est + args.lambda_fair * loss_attack + 0.5 * loss_util
        
        opt_est.zero_grad()
        opt_sur.zero_grad()
        opt_atk.zero_grad()
        total_loss.backward()
        opt_est.step()
        opt_sur.step()
        opt_atk.step()
        
        # === 日志打印 ===
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                # 简单评估 Estimator 准确率
                est_acc, est_auc = evaluate_performance(s_logits, S_clean)
                print(f"Epoch {epoch+1:03d} | Total: {total_loss.item():.2f} | "
                      f"MI(Z;S): {mutual_info_proxy.item():.4f} | "
                      f"Est Acc: {est_acc:.4f} | "
                      f"Attack(SP): {-loss_sp.item():.4f} | Attack(EO): {-loss_eo.item():.4f}")

    # ------------------------------------------------------------------
    # 5. 导出最终毒药图 (离散化)
    # ------------------------------------------------------------------
    print("\n[COFA] Generating Discrete Poisoned Graph...")
    with torch.no_grad():
        final_scores = base_w + P
        
        # 调试信息
        probs = torch.sigmoid(final_scores)
        print("-" * 30)
        print("[DEBUG] P (Soft Weights) Stats:")
        print(f"  Max: {probs.max().item():.4f}, Min: {probs.min().item():.4f}, Mean: {probs.mean().item():.4f}")
        print("-" * 30)

        # Top-K 选择
        current_num_edges = pos_edge_index.shape[1]
        target_budget = int(current_num_edges * (1 + args.ptb_rate)) 
        budget_edges = min(target_budget, final_scores.shape[0])
        
        _, top_indices = torch.topk(final_scores, budget_edges)
        final_edge_index = candidate_edge_index[:, top_indices]
        
        save_dir = f'./data/{args.dataset}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = f'{save_dir}/COFA_poisoned_{args.surrogate_model}_seed{args.seed}.pt'
        torch.save(final_edge_index, save_path)
        
        print(f"Saved poisoned adjacency to: {save_path}")

if __name__ == "__main__":
    train_cofa()