import subprocess
import pandas as pd
import itertools
import sys
import os
import time

# ================= 配置区域 =================

# 1. 全量数据集列表
datasets = ['pokec_z', 'pokec_n', 'credit', 'german', 'dblp', 'bail']

# 2. 全量模型列表
# 注意: FairGNN 运行速度可能较慢，且需要你的 model.py 支持该架构
models = ['GCN', 'GraphSAGE', 'APPNP', 'FairGNN']

# 3. 随机种子 (建议跑 3-5 个，这里为了演示设为 5 个)
seeds = [0, 1, 2, 3, 4]

# 4. 指定 GPU
device = "cuda:0" 

# 5. 通用参数
common_args = {
    'epochs': 500,          # 攻击生成器训练轮数
    'ptb_rate': 0.05,       # 扰动率 (5%)
    'noise_rate': 0.1,      # LDP 噪声率
    'lambda_fair': 1.0,     # 攻击力度
    # 'victim_epochs': 200  # 受害者重训练轮数 (默认200)
}

# ===========================================

def run_experiments():
    # 生成所有任务组合
    configs = list(itertools.product(datasets, models, seeds))
    total_tasks = len(configs)
    results = []

    print(f"🚀 [COFA Full-Batch Runner] 开始全量实验，总计 {total_tasks} 个任务")
    print(f"📌 配置: Device={device} | Noise={common_args['noise_rate']} | Ptb={common_args['ptb_rate']}")
    print("-" * 100)
    # 打印表头
    print(f"{'Dataset':<10} | {'Model':<10} | {'Seed':<4} | {'Status':<10} | {'SP (Clean->Pois)':<20} | {'EO (Clean->Pois)':<20}")
    print("-" * 100)

    start_time_all = time.time()

    for idx, (dataset, model, seed) in enumerate(configs):
        # 构造命令
        cmd = [
            sys.executable, "main.py",
            "--dataset", dataset,
            "--surrogate_model", model,
            "--seed", str(seed),
            "--device", device
        ]
        for k, v in common_args.items():
            cmd.extend([f"--{k}", str(v)])
            
        try:
            # 运行任务
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                print(f"{dataset:<10} | {model:<10} | {seed:<4} | ❌ Fail     | {'Check Log':<20} | {'Check Log':<20}")
                # 将错误写入单独的日志文件
                with open("error_log.txt", "a") as f:
                    f.write(f"\n=== Error in {dataset} {model} {seed} ===\n")
                    f.write(result.stderr[-500:]) # 只记录最后500字符
                continue
            
            # 解析输出
            parsed = False
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.startswith("FINAL_RESULT"):
                    # 格式: FINAL_RESULT,dataset,model,seed,c_acc,c_sp,c_eo,p_acc,p_sp,p_eo
                    parts = line.strip().split(',')
                    res = {
                        'dataset': parts[1],
                        'model': parts[2],
                        'seed': int(parts[3]),
                        'clean_acc': float(parts[4]),
                        'clean_sp': float(parts[5]),
                        'clean_eo': float(parts[6]),
                        'poison_acc': float(parts[7]),
                        'poison_sp': float(parts[8]),
                        'poison_eo': float(parts[9])
                    }
                    results.append(res)
                    
                    # 格式化输出字符串
                    sp_change = f"{res['clean_sp']:.3f} -> {res['poison_sp']:.3f}"
                    eo_change = f"{res['clean_eo']:.3f} -> {res['poison_eo']:.3f}"
                    
                    print(f"{dataset:<10} | {model:<10} | {seed:<4} | ✅ Done     | {sp_change:<20} | {eo_change:<20}")
                    parsed = True
                    break
            
            if not parsed:
                print(f"{dataset:<10} | {model:<10} | {seed:<4} | ⚠️ No Res   | {'Unknown':<20} | {'Unknown':<20}")

        except Exception as e:
            print(f"System Error: {e}")

    # ================= 结果统计 =================
    total_time = time.time() - start_time_all
    print("-" * 100)
    print(f"🏁 实验结束，耗时: {total_time/60:.2f} 分钟")

    if results:
        df = pd.DataFrame(results)
        
        # 1. 保存原始数据
        df.to_csv("cofa_results_raw.csv", index=False)
        print(f"💾 原始数据: cofa_results_raw.csv")
        
        # 2. 计算均值 (Mean)
        summary = df.groupby(['dataset', 'model'])[[
            'clean_sp', 'poison_sp', 
            'clean_eo', 'poison_eo', 
            'clean_acc', 'poison_acc'
        ]].mean()
        
        # 3. 保存汇总表
        summary.to_csv("cofa_results_summary.csv")
        print(f"📊 汇总统计: cofa_results_summary.csv")
        
        # 4. 打印高亮对比表
        print("\n=== 最终效果汇总 (Mean) ===")
        # 选取最重要的列进行打印
        print(summary[['clean_sp', 'poison_sp', 'clean_eo', 'poison_eo']])
    else:
        print("❌ 未收集到任何数据。")

if __name__ == "__main__":
    run_experiments()