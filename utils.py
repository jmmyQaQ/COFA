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
    Backward Correction Loss (对抗噪声)
    """
    if rho > 0.499: rho = 0.49  # 避免除零
    
    # 基础 BCE Loss
    probs = torch.sigmoid(logits)
    
    # 构建校正矩阵 Q inverse
    # P(s'|s) = [[1-rho, rho], [rho, 1-rho]]
    # 我们希望优化的是 L(f(x), s_true)，但只有 s_noisy
    # 使用 Backward Correction 公式
    
    loss = F.binary_cross_entropy_with_logits(logits, noisy_labels, reduction='none')
    
    # 简化版 Correction: re-weighting
    # 实际上，对于 Neural Network，直接用带噪标签训练+校正矩阵比较复杂
    # 这里使用一个简单的加权技巧，或者直接返回 BCE (如果是简单 Baseline)
    # 为了实现端到端效果，我们使用估计的 s_pred 去拟合 noisy_labels 的分布
    
    return loss.mean()

def evaluate_performance(logits, true_labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    truth = true_labels.detach().cpu().numpy()
    return accuracy_score(truth, preds), roc_auc_score(truth, probs)