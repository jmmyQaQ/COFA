import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
from dataset import load_fairness_dataset
from config import get_args
from model import SensitiveEstimator, SurrogateModel
from utils import backward_correction_loss, apply_ldp_noise, estimate_noise_parameter

# ... (保留 set_seed, ensure_masks, gaussian_kernel, estimate_density, evaluate_metrics 等原有工具函数不变) ...
# 为了节省篇幅，这里假设上方所有工具函数已保留，仅展示修改后的 train_cofa 和 test_victim_performance

def ensure_masks(data, device):
    """确保所有数据集都有 train/val/test mask。"""
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes, device=device)
        train_size = int(0.5 * num_nodes)
        val_size = int(0.25 * num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size:train_size+val_size]] = True
        data.test_mask[indices[train_size+val_size:]] = True
    return data

# [工具函数] 简单的 Gaussian Kernel (同原文件)
def gaussian_kernel(x, mu, sigma=0.05):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * 3.1415926535)**0.5)

# [工具函数] KDE (同原文件)
def estimate_density(y_preds, weights, sample_points, sigma=0.05):
    y_preds_exp = y_preds.unsqueeze(1)      
    sample_points_exp = sample_points.unsqueeze(0)
    kernels = gaussian_kernel(sample_points_exp, y_preds_exp, sigma)
    weights_norm = weights / (weights.sum() + 1e-6)
    density = (kernels * weights_norm.unsqueeze(1)).sum(dim=0)
    return density

def test_victim_performance(data, adj_to_test, args, verbose=False):
    # 初始化受害者模型 (传入 GRASP 参数)
    victim_model = SurrogateModel(
        args.surrogate_model, data.num_features, args.hidden_dim, int(data.y.max().item())+1,
        enable_grasp=args.enable_grasp, grasp_sigma=args.grasp_sigma, grasp_p=args.grasp_p
    ).to(args.device)
    
    optimizer = optim.Adam(victim_model.parameters(), lr=args.lr_sur, weight_decay=1e-5)
    
    victim_model.train()
    # 正常训练受害者 (受害者会在训练时开启 GRASP 防御)
    for epoch in range(200):
        optimizer.zero_grad()
        output = victim_model(data.x, adj_to_test, None) 
        loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
    victim_model.eval()
    with torch.no_grad():
        # 推理时是否开启 GRASP? 
        # 论文中 GRASP 在 inference 阶段也提供保护 [cite: 348]
        # 但通常评估 utility 时我们可以看 clean output，这里为了模拟真实防御环境，保持 GRASP 开启
        output = victim_model(data.x, adj_to_test, None)
        # 简单计算 Metric
        pred = output.argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
        
        # 计算 SP
        sens = data.sens
        mask = data.test_mask
        pred_test = pred[mask]
        sens_test = sens[mask]
        idx_s0 = (sens_test == 0)
        idx_s1 = (sens_test == 1)
        rate_s0 = (pred_test[idx_s0] == 1).float().mean().item() if idx_s0.sum() > 0 else 0
        rate_s1 = (pred_test[idx_s1] == 1).float().mean().item() if idx_s1.sum() > 0 else 0
        sp_gap = abs(rate_s1 - rate_s0)
        
        # 计算 EO
        y_test = data.y[mask]
        idx_y1 = (y_test == 1)
        idx_s0_y1 = idx_s0 & idx_y1
        idx_s1_y1 = idx_s1 & idx_y1
        tpr_s0 = (pred_test[idx_s0_y1] == 1).float().mean().item() if idx_s0_y1.sum() > 0 else 0
        tpr_s1 = (pred_test[idx_s1_y1] == 1).float().mean().item() if idx_s1_y1.sum() > 0 else 0
        eo_gap = abs(tpr_s1 - tpr_s0)

    return acc, sp_gap, eo_gap

def train_cofa():
    args = get_args()
    device = torch.device(args.device)
    # set_seed(args.seed) # 保持随机性以便 EoT 采样
    
    print(f"=== Config: GRASP={args.enable_grasp} (sigma={args.grasp_sigma}, p={args.grasp_p}) | LDP_Noise={args.noise_rate} ===")

    # 1. 加载数据
    try:
        data = load_fairness_dataset(args.dataset, args.data_path).to(device)
    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        return

    data = ensure_masks(data, device) 
    if len(torch.unique(data.sens)) > 2:
        data.sens = (data.sens > 0).float() 

    # 类别权重
    y_train = data.y[data.train_mask]
    class_counts = torch.bincount(y_train) + 1
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)

    # LDP 环境模拟
    S_clean = data.sens
    S_noisy = apply_ldp_noise(S_clean, args.noise_rate).to(device)
    
    # 攻击者估算 rho
    attacker_prior = 0.5 
    if args.dataset.lower() in ['german', 'bail']: attacker_prior = 0.7
    estimated_rho = estimate_noise_parameter(S_noisy, prior_pos_ratio=attacker_prior)
    kde_sample_points = torch.linspace(0, 1, 100, device=device)

    # 阶段 0: 攻击前基准
    clean_acc, clean_sp, clean_eo = test_victim_performance(data, data.edge_index, args)

    # 3. 初始化组件
    estimator = SensitiveEstimator(data.num_features, args.hidden_dim).to(device)
    opt_est = optim.Adam(estimator.parameters(), lr=args.lr_est)

    # 受害者模型 (包含 GRASP 设置)
    surrogate = SurrogateModel(
        args.surrogate_model, data.num_features, args.hidden_dim, int(data.y.max().item())+1,
        enable_grasp=args.enable_grasp, grasp_sigma=args.grasp_sigma, grasp_p=args.grasp_p
    ).to(device)
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
        
        # A. 构图 (Gumbel Softmax)
        logits = base_w + P
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        edge_weight_soft = torch.sigmoid((logits + gumbel_noise) / 1.0)
        
        # B. Estimator Update (不变)
        s_logits = estimator(data.x, candidate_edge_index, edge_weight_soft)
        loss_est = backward_correction_loss(s_logits.squeeze(), S_noisy, estimated_rho)
        
        opt_est.zero_grad()
        loss_est.backward(retain_graph=True) # 因为 edge_weight_soft 还要用
        opt_est.step()

        # === 核心修改: Expectation over Transformation (EoT) 用于 Surrogate & Attack ===
        # 如果开启了 GRASP，我们需要对受害者的 forward pass 进行多次采样，以获得稳定的攻击梯度
        
        loss_sur_total = 0
        loss_atk_total = 0
        
        # 确定采样次数：如果是 GRASP 开启，则采样 K 次；否则 1 次
        n_samples = args.atk_eot_samples if args.enable_grasp else 1
        
        # 由于 edge_weight_soft 是一次生成的，我们在计算 surrogate loss 时
        # 需要让 surrogate 内部的 dropout 和 GRASP noise 多次随机
        
        for _ in range(n_samples):
            # C. Surrogate Forward (带 GRASP 噪声)
            y_logits = surrogate(data.x, candidate_edge_index, edge_weight_soft)
            loss_util = F.cross_entropy(y_logits[data.train_mask], data.y[data.train_mask], weight=class_weights)
            
            # D. Fairness Attack Loss Calculation
            s_probs_detached = torch.sigmoid(s_logits).squeeze().detach()
            prob_s1 = s_probs_detached
            prob_s0 = 1 - s_probs_detached
            
            # 预测概率
            y_pred_prob = F.softmax(y_logits, dim=1)[:, 1] if y_logits.shape[1] > 1 else torch.sigmoid(y_logits).squeeze()

            # Wasserstein Loss
            pdf_s1 = estimate_density(y_pred_prob, prob_s1, kde_sample_points, sigma=0.05)
            pdf_s0 = estimate_density(y_pred_prob, prob_s0, kde_sample_points, sigma=0.05)
            cdf_s1 = torch.cumsum(pdf_s1, dim=0) / (torch.cumsum(pdf_s1, dim=0)[-1] + 1e-6)
            cdf_s0 = torch.cumsum(pdf_s0, dim=0) / (torch.cumsum(pdf_s0, dim=0)[-1] + 1e-6)
            dist_wasserstein = torch.sum(torch.abs(cdf_s1 - cdf_s0))
            
            # Directional Loss
            mean_pred_s1 = (y_pred_prob * prob_s1).sum() / (prob_s1.sum() + 1e-6)
            mean_pred_s0 = (y_pred_prob * prob_s0).sum() / (prob_s0.sum() + 1e-6)
            loss_discrimination = mean_pred_s1 - mean_pred_s0
            
            loss_attack_sample = - dist_wasserstein + 1.0 * loss_discrimination
            
            # 累加 Loss
            loss_sur_total += loss_util
            loss_atk_total += (args.lambda_fair * loss_attack_sample + 0.5 * loss_util)

        # 取平均
        loss_sur_avg = loss_sur_total / n_samples
        loss_atk_avg = loss_atk_total / n_samples
        
        # 更新参数
        opt_sur.zero_grad()
        loss_sur_avg.backward(retain_graph=True)
        opt_sur.step()
        
        opt_atk.zero_grad()
        loss_atk_avg.backward()
        opt_atk.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss Est={loss_est.item():.4f}, Loss Sur={loss_sur_avg.item():.4f}, Loss Atk={loss_atk_avg.item():.4f}")

    # 6. 生成毒药图
    with torch.no_grad():
        final_scores = base_w + P
        target_budget = int(data.edge_index.shape[1] * (1 + args.ptb_rate)) 
        budget_edges = min(target_budget, final_scores.shape[0])
        _, top_indices = torch.topk(final_scores, budget_edges)
        final_edge_index = candidate_edge_index[:, top_indices]
        
        # 保存中毒的边，以便 evaluate_attack.py 使用
        save_path = f'./data/{args.dataset}/COFA_poisoned_{args.surrogate_model}_seed{args.seed}.pt'
        torch.save(final_edge_index, save_path)

    # 7. 攻击后评估
    # 注意：这里评估时也应该开启 GRASP，看看在防御下的表现
    poison_acc, poison_sp, poison_eo = test_victim_performance(data, final_edge_index, args)
    
    print(f"FINAL_RESULT,{args.dataset},{args.surrogate_model},{args.seed},{clean_acc:.4f},{clean_sp:.4f},{clean_eo:.4f},{poison_acc:.4f},{poison_sp:.4f},{poison_eo:.4f}")

if __name__ == "__main__":
    train_cofa()