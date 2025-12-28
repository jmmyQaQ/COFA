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

def backward_correction_loss(logits, noisy_labels, rho):
    """
    [修复版] Backward Correction Loss (对抗噪声)
    
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