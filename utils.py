import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

def apply_ldp_noise(sens_true, rho):
    """
    对敏感属性加噪 (Flip Noise)
    """
    if rho == 0: return sens_true.clone()
    
    n = sens_true.shape[0]
    mask = torch.rand(n, device=sens_true.device) < rho
    sens_noisy = sens_true.clone()
    # 0->1, 1->0
    sens_noisy[mask] = 1 - sens_noisy[mask]
    return sens_noisy

def estimate_noise_parameter(noisy_sens, prior_pos_ratio=0.5):
    """
    [新增模块] 攻击者自适应参数估算
    根据观测到的噪声数据分布和先验知识，反推隐私保护参数 rho。
    
    原理: Method of Moments (矩估计)
    E[S_noisy] = prior * (1-rho) + (1-prior) * rho
    """
    # 1. 计算观测到的 S=1 的比例
    observed_mean = noisy_sens.float().mean().item()
    
    # 2. 避免先验过于极端导致分母为0
    if abs(1 - 2 * prior_pos_ratio) < 1e-4:
        # 如果先验认为正负样本完全各占一半(0.5)，且观测值也是0.5，则无法区分是否有噪声
        # 此时保守估计 rho=0 (或者认为无攻击空间)
        return 0.0
        
    # 3. 根据公式反推 rho
    # rho = (mean_obs - prior) / (1 - 2*prior)
    rho_est = (observed_mean - prior_pos_ratio) / (1 - 2 * prior_pos_ratio)
    
    # 4. 约束 rho 在合理范围内 [0, 0.5)
    # 哪怕估算有误差，截断操作能保证数学计算的稳定性
    rho_est = max(0.0, min(rho_est, 0.499))
    
    print(f"\n[Attacker Knowledge] Observed Mean: {observed_mean:.4f}, Prior: {prior_pos_ratio}")
    print(f"[Attacker Knowledge] Estimated Privacy Parameter (rho): {rho_est:.4f} (True rho might be unknown)\n")
    
    return rho_est

def backward_correction_loss(logits, noisy_labels, rho):
    """
    Backward Correction Loss (对抗噪声)
    
    原理: 构造无偏估计量，使得 E[Loss_corrected] = Loss_true
    公式: L_unbiased = [ (1-rho) * L(y_noisy) - rho * L(1-y_noisy) ] / (1 - 2*rho)
    """
    # 1. 边界保护，防止除零 (rho 通常 < 0.5)
    rho = min(rho, 0.499)
    
    # 2. 计算基于当前 noisy_labels 的损失 (Observed Loss)
    loss_observed = F.binary_cross_entropy_with_logits(logits, noisy_labels, reduction='none')
    
    # 3. 计算基于反转标签的损失 (Flipped Loss)
    loss_flipped = F.binary_cross_entropy_with_logits(logits, 1 - noisy_labels, reduction='none')
    
    # 4. 应用 Backward Correction 公式进行校正
    corrected_loss = ((1 - rho) * loss_observed - rho * loss_flipped) / (1 - 2 * rho)
    
    return corrected_loss.mean()

def evaluate_performance(logits, true_labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    truth = true_labels.detach().cpu().numpy()
    return accuracy_score(truth, preds), roc_auc_score(truth, probs)