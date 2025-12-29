import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from dataset import load_fairness_dataset
from model import SurrogateModel
from config import get_args
from sklearn.metrics import accuracy_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_fairness_metrics(preds, labels, sens):
    """
    计算 Accuracy, Statistical Parity (SP), Equal Opportunity (EO)
    """
    # 1. Accuracy
    acc = accuracy_score(labels.cpu(), preds.cpu())
    
    # 准备 Mask
    s1_mask = (sens == 1)
    s0_mask = (sens == 0)
    y1_mask = (labels == 1) # 真实标签为正例 (用于计算 EO)
    
    # 2. Statistical Parity Gap (SP)
    # P(pred=1 | s=1) vs P(pred=1 | s=0)
    rate_s1 = (preds[s1_mask] == 1).float().mean().item() if s1_mask.sum() > 0 else 0.0
    rate_s0 = (preds[s0_mask] == 1).float().mean().item() if s0_mask.sum() > 0 else 0.0
    sp_gap = abs(rate_s1 - rate_s0)
    
    # 3. Equal Opportunity Gap (EO)
    # P(pred=1 | y=1, s=1) vs P(pred=1 | y=1, s=0) -> 即 TPR 的差异
    s1_y1_mask = s1_mask & y1_mask
    s0_y1_mask = s0_mask & y1_mask
    
    tpr_s1 = (preds[s1_y1_mask] == 1).float().mean().item() if s1_y1_mask.sum() > 0 else 0.0
    tpr_s0 = (preds[s0_y1_mask] == 1).float().mean().item() if s0_y1_mask.sum() > 0 else 0.0
    eo_gap = abs(tpr_s1 - tpr_s0)
    
    return acc, sp_gap, eo_gap

def train_and_eval(data, args, seed, edge_index_override=None):
    set_seed(seed) 
    
    device = torch.device(args.device)
    # 根据数据集类别数动态调整输出维度
    num_classes = int(data.y.max().item()) + 1
    model = SurrogateModel(args.surrogate_model, data.num_features, args.hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_sur, weight_decay=1e-5)
    
    edge_index = edge_index_override if edge_index_override is not None else data.edge_index
    edge_index = edge_index.to(device)
    x, y, sens = data.x.to(device), data.y.to(device), data.sens.to(device)
    
    torch.manual_seed(seed) 
    indices = torch.randperm(data.num_nodes)
    n_train = int(data.num_nodes * 0.6)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        preds = logits.argmax(dim=1)
        
        # 只在测试集上计算指标
        acc, sp, eo = compute_fairness_metrics(preds[test_idx], y[test_idx], sens[test_idx])
        
    return acc, sp, eo

def main():
    args = get_args()
    base_seed = args.seed if args.seed else 42
    # 评估5个种子
    eval_seeds = [base_seed, base_seed+1, base_seed+2, base_seed+3, base_seed+4]
    
    print(f"Dataset: {args.dataset} | Model: {args.surrogate_model}")
    data = load_fairness_dataset(args.dataset, args.data_path)
    
    # === 1. 评估原始图 ===
    print("\n" + "-"*40)
    print(">>> Evaluating on CLEAN Graph...")
    clean_results = {'acc': [], 'sp': [], 'eo': []}
    
    for s in eval_seeds:
        acc, sp, eo = train_and_eval(data, args, seed=s, edge_index_override=None)
        clean_results['acc'].append(acc)
        clean_results['sp'].append(sp)
        clean_results['eo'].append(eo)
        print(f"   Seed {s}: Acc={acc:.4f}, SP={sp:.4f}, EO={eo:.4f}")
    
    # === 2. 评估毒药图 ===
    poison_path_seeded = f'./data/{args.dataset}/COFA_poisoned_{args.surrogate_model}_seed{base_seed}.pt'
    poison_path_generic = f'./data/{args.dataset}/COFA_poisoned_{args.surrogate_model}.pt'
    
    if os.path.exists(poison_path_seeded):
        poison_path = poison_path_seeded
    elif os.path.exists(poison_path_generic):
        poison_path = poison_path_generic
    else:
        print(f"\n[Error] Poisoned file not found: {poison_path_seeded}")
        return

    print("\n" + "-"*40)
    print(f">>> Evaluating on POISONED Graph ({os.path.basename(poison_path)})...")
    
    poison_edge_index = torch.load(poison_path)
    poison_results = {'acc': [], 'sp': [], 'eo': []}
    
    for s in eval_seeds:
        acc, sp, eo = train_and_eval(data, args, seed=s, edge_index_override=poison_edge_index)
        poison_results['acc'].append(acc)
        poison_results['sp'].append(sp)
        poison_results['eo'].append(eo)
        print(f"   Seed {s}: Acc={acc:.4f}, SP={sp:.4f}, EO={eo:.4f}")

    # === 3. 计算统计指标 (Average & Min-Max) ===
    
    # A. 平均值 (Standard)
    avg_clean_acc = np.mean(clean_results['acc'])
    avg_poison_acc = np.mean(poison_results['acc'])
    
    avg_clean_sp = np.mean(clean_results['sp'])
    avg_poison_sp = np.mean(poison_results['sp'])
    
    avg_clean_eo = np.mean(clean_results['eo'])
    avg_poison_eo = np.mean(poison_results['eo'])

    # B. 极值对比 (Superiority Analysis)
    min_clean_sp = np.min(clean_results['sp'])
    max_poison_sp = np.max(poison_results['sp'])
    
    min_clean_eo = np.min(clean_results['eo'])
    max_poison_eo = np.max(poison_results['eo'])

    print("\n" + "="*60)
    print("FINAL RESULTS REPORT")
    print("="*60)
    
    # 表头 (之前报错的那一行，现在修复了)
    print(f"{'Metric Strategy':<20} | {'Metric':<5} | {'Clean':<8} -> {'Poisoned':<8} | {'Delta':<8}")
    print("-" * 60)
    
    # 1. Average Comparison
    print(f"{'Average (Std)':<20} | {'Acc':<5} | {avg_clean_acc:.4f} -> {avg_poison_acc:.4f}   | {avg_poison_acc - avg_clean_acc:+.4f}")
    print(f"{'':<20} | {'SP':<5} | {avg_clean_sp:.4f} -> {avg_poison_sp:.4f}   | {avg_poison_sp - avg_clean_sp:+.4f}")
    print(f"{'':<20} | {'EO':<5} | {avg_clean_eo:.4f} -> {avg_poison_eo:.4f}   | {avg_poison_eo - avg_clean_eo:+.4f}")
    
    print("-" * 60)
    
    # 2. Min-Max Comparison
    print(f"{'Min-Clean vs Max-Ptb':<20} | {'SP':<5} | {min_clean_sp:.4f} -> {max_poison_sp:.4f}   | {max_poison_sp - min_clean_sp:+.4f}")
    print(f"{'':<20} | {'EO':<5} | {min_clean_eo:.4f} -> {max_poison_eo:.4f}   | {max_poison_eo - min_clean_eo:+.4f}")
    print("="*60)

if __name__ == "__main__":
    main()