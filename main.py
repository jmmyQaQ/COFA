import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
from dataset import load_fairness_dataset
from config import get_args
from model import SensitiveEstimator, SurrogateModel
from utils import backward_correction_loss, apply_ldp_noise

# ==============================================================================
# 工具函数
# ==============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_masks(data, device):
    """
    确保所有数据集都有 train/val/test mask。
    对于 bail, credit, german, dblp 等可能缺失 mask 的数据集进行随机划分。
    """
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes, device=device)
        
        # 划分比例 50% Train, 25% Val, 25% Test
        train_size = int(0.5 * num_nodes)
        val_size = int(0.25 * num_nodes)
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size:train_size+val_size]] = True
        data.test_mask[indices[train_size+val_size:]] = True
    
    return data

def gaussian_kernel(x, mu, sigma=0.05):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * 3.1415926535)**0.5)

def estimate_density(y_preds, weights, sample_points, sigma=0.05):
    y_preds_exp = y_preds.unsqueeze(1)      
    sample_points_exp = sample_points.unsqueeze(0)
    kernels = gaussian_kernel(sample_points_exp, y_preds_exp, sigma)
    weights_norm = weights / (weights.sum() + 1e-6)
    density = (kernels * weights_norm.unsqueeze(1)).sum(dim=0)
    return density

# ==============================================================================
# 评估模块 (Evaluation Module)
# ==============================================================================

def evaluate_metrics(output, labels, sens, mask):
    """计算 ACC, SP, EO"""
    pred = output.argmax(dim=1)
    
    # 1. Accuracy
    correct = pred[mask].eq(labels[mask]).sum().item()
    acc = correct / mask.sum().item()
    
    # 2. Fairness Metrics
    pred_test = pred[mask]
    sens_test = sens[mask]
    y_test = labels[mask]
    target_class = 1 
    
    idx_s0 = (sens_test == 0)
    idx_s1 = (sens_test == 1)
    
    # SP
    rate_s0 = (pred_test[idx_s0] == target_class).float().mean().item() if idx_s0.sum() > 0 else 0
    rate_s1 = (pred_test[idx_s1] == target_class).float().mean().item() if idx_s1.sum() > 0 else 0
    sp_gap = abs(rate_s1 - rate_s0)
    
    # EO (TPR Gap + FPR Gap)
    idx_y1 = (y_test == 1)
    idx_s0_y1 = idx_s0 & idx_y1
    idx_s1_y1 = idx_s1 & idx_y1
    tpr_s0 = (pred_test[idx_s0_y1] == 1).float().mean().item() if idx_s0_y1.sum() > 0 else 0
    tpr_s1 = (pred_test[idx_s1_y1] == 1).float().mean().item() if idx_s1_y1.sum() > 0 else 0
    tpr_gap = abs(tpr_s1 - tpr_s0)
    
    idx_y0 = (y_test == 0)
    idx_s0_y0 = idx_s0 & idx_y0
    idx_s1_y0 = idx_s1 & idx_y0
    fpr_s0 = (pred_test[idx_s0_y0] == 1).float().mean().item() if idx_s0_y0.sum() > 0 else 0
    fpr_s1 = (pred_test[idx_s1_y0] == 1).float().mean().item() if idx_s1_y0.sum() > 0 else 0
    fpr_gap = abs(fpr_s1 - fpr_s0)
    
    eo_gap = tpr_gap + fpr_gap
    
    return acc, sp_gap, eo_gap

def test_victim_performance(data, adj_to_test, args, verbose=False):
    # 初始化受害者模型
    # 注意: FairGNN 通常需要特殊参数，这里假设 model.py 中的 SurrogateModel 做了兼容封装
    # 如果 FairGNN 需要 sens 输入，请确保 model.py 的 forward 能够处理
    victim_model = SurrogateModel(args.surrogate_model, data.num_features, args.hidden_dim, int(data.y.max().item())+1).to(args.device)
    optimizer = optim.Adam(victim_model.parameters(), lr=args.lr_sur, weight_decay=args.weight_decay)
    
    victim_model.train()
    train_epochs = getattr(args, 'victim_epochs', 200)
    
    for epoch in range(train_epochs):
        optimizer.zero_grad()
        # 这里传入 None 作为 edge_weight，因为 adj_to_test 是离散的 edge_index
        output = victim_model(data.x, adj_to_test, None) 
        loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
    victim_model.eval()
    with torch.no_grad():
        output = victim_model(data.x, adj_to_test, None)
        acc, sp_gap, eo_gap = evaluate_metrics(output, data.y, data.sens, data.test_mask)
    
    return acc, sp_gap, eo_gap

# ==============================================================================
# 主训练逻辑
# ==============================================================================

def train_cofa():
    args = get_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    
    # 1. 加载数据
    try:
        data = load_fairness_dataset(args.dataset, args.data_path).to(device)
    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        return

    # 2. 预处理
    data = ensure_masks(data, device) # 确保有mask
    if len(torch.unique(data.sens)) > 2:
        data.sens = (data.sens > 0).float() # 二值化敏感属性

    # 计算类别权重
    y_train = data.y[data.train_mask]
    if y_train.shape[0] == 0: # 防止空训练集报错
        class_weights = torch.ones(int(data.y.max().item())+1).to(device)
    else:
        class_counts = torch.bincount(y_train) + 1
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = class_weights.to(device)

    # LDP 噪声
    S_clean = data.sens
    S_noisy = apply_ldp_noise(S_clean, args.noise_rate).to(device)
    kde_sample_points = torch.linspace(0, 1, 100, device=device)

    # ----------------------------------------------------
    # 阶段 0: 攻击前基准 (Clean Baseline)
    # ----------------------------------------------------
    clean_acc, clean_sp, clean_eo = test_victim_performance(data, data.edge_index, args)

    # 3. 初始化组件
    estimator = SensitiveEstimator(data.num_features, args.hidden_dim).to(device)
    opt_est = optim.Adam(estimator.parameters(), lr=args.lr_est)

    surrogate = SurrogateModel(args.surrogate_model, data.num_features, args.hidden_dim, int(data.y.max().item())+1).to(device)
    opt_sur = optim.Adam(surrogate.parameters(), lr=args.lr_sur)

    # 4. 初始化攻击变量 P
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    
    # 负采样
    min_neg_needed = int(num_edges * args.ptb_rate * 1.5)
    num_neg_samples = min(num_nodes * num_nodes, max(200000, min_neg_needed))
    num_neg_samples = min(num_neg_samples, 2000000)
    
    neg_row = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    neg_col = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
    candidate_edge_index = torch.cat([data.edge_index, neg_edge_index], dim=1)
    
    base_w = torch.cat([torch.ones(data.edge_index.shape[1], device=device),
                        torch.zeros(neg_edge_index.shape[1], device=device)])
    
    P = torch.zeros(candidate_edge_index.shape[1], device=device, requires_grad=True)
    opt_atk = optim.Adam([P], lr=args.lr_atk)

    # 5. 联合训练
    for epoch in range(args.epochs):
        estimator.train()
        surrogate.train()
        
        # A. 构图
        logits = base_w + P
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        edge_weight_soft = torch.sigmoid((logits + gumbel_noise) / 1.0)
        
        # B. Estimator Update
        s_logits = estimator(data.x, candidate_edge_index, edge_weight_soft)
        loss_est = backward_correction_loss(s_logits.squeeze(), S_noisy, args.noise_rate)

        # C. Surrogate Update
        y_logits = surrogate(data.x, candidate_edge_index, edge_weight_soft)
        loss_util = F.cross_entropy(y_logits[data.train_mask], data.y[data.train_mask], weight=class_weights)
        
        # D. Wasserstein Attack
        s_probs_detached = torch.sigmoid(s_logits).squeeze().detach()
        prob_s1 = s_probs_detached
        prob_s0 = 1 - s_probs_detached
        
        y_pred_prob = F.softmax(y_logits, dim=1)[:, 1] if y_logits.shape[1] > 1 else torch.sigmoid(y_logits).squeeze()

        pdf_s1 = estimate_density(y_pred_prob, prob_s1, kde_sample_points, sigma=0.05)
        pdf_s0 = estimate_density(y_pred_prob, prob_s0, kde_sample_points, sigma=0.05)
        
        cdf_s1 = torch.cumsum(pdf_s1, dim=0)
        cdf_s0 = torch.cumsum(pdf_s0, dim=0)
        cdf_s1 = cdf_s1 / (cdf_s1[-1] + 1e-6)
        cdf_s0 = cdf_s0 / (cdf_s0[-1] + 1e-6)
        
        dist_wasserstein = torch.sum(torch.abs(cdf_s1 - cdf_s0))
        loss_attack = - dist_wasserstein
        
        total_loss = loss_est + args.lambda_fair * loss_attack + 0.5 * loss_util
        
        opt_est.zero_grad()
        opt_sur.zero_grad()
        opt_atk.zero_grad()
        total_loss.backward()
        opt_est.step()
        opt_sur.step()
        opt_atk.step()

    # 6. 生成毒药图
    with torch.no_grad():
        final_scores = base_w + P
        target_budget = int(data.edge_index.shape[1] * (1 + args.ptb_rate)) 
        budget_edges = min(target_budget, final_scores.shape[0])
        _, top_indices = torch.topk(final_scores, budget_edges)
        final_edge_index = candidate_edge_index[:, top_indices]

    # 7. 攻击后评估 (Poisoned Evaluation)
    poison_acc, poison_sp, poison_eo = test_victim_performance(data, final_edge_index, args)
    
    # 格式: FINAL_RESULT,dataset,model,seed,clean_acc,clean_sp,clean_eo,poison_acc,poison_sp,poison_eo
    print(f"FINAL_RESULT,{args.dataset},{args.surrogate_model},{args.seed},{clean_acc:.4f},{clean_sp:.4f},{clean_eo:.4f},{poison_acc:.4f},{poison_sp:.4f},{poison_eo:.4f}")

if __name__ == "__main__":
    train_cofa()