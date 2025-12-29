#!/bin/bash

# ==============================================================================
# COFA Full Factorial Benchmark Script
# Target: 6 Datasets x 4 Models = 24 Experiments
# ==============================================================================

# 创建数据保存目录
mkdir -p ./data/german ./data/bail ./data/credit ./data/pokec_z ./data/pokec_n ./data/dblp
mkdir -p ./logs

# 设置 GPU (根据实际情况修改)
export CUDA_VISIBLE_DEVICES=0

# 定义模型列表
MODELS=("GCN" "GraphSAGE" "FairGNN" "APPNP")

# 定义数据集列表
DATASETS=("german" "bail" "credit" "pokec_z" "pokec_n" "dblp")

# ==============================================================================
# 函数: 运行单个实验
# ==============================================================================
run_experiment() {
    DATASET=$1
    MODEL=$2
    
    # --- 1. 根据数据集自动配置参数 (可根据需要微调) ---
    case $DATASET in
        "german")
            # 小图：需要高 Budget 和适中 Lambda
            EPOCHS=60
            LAMBDA=5.0
            BUDGET=0.15
            LR_ATK=0.1
            HIDDEN=32
            ;;
        "bail")
            # 中图：标准配置
            EPOCHS=80
            LAMBDA=8.0
            BUDGET=0.05
            LR_ATK=0.05
            HIDDEN=32
            ;;
        "credit")
            # 大图：高 Lambda，较大 Budget
            EPOCHS=100
            LAMBDA=15.0
            BUDGET=0.10
            LR_ATK=0.05
            HIDDEN=64
            ;;
        "pokec_z" | "pokec_n")
            # 特大图：高 Lambda，标准 Budget (防止OOM)
            EPOCHS=100
            LAMBDA=20.0
            BUDGET=0.05
            LR_ATK=0.02
            HIDDEN=64
            ;;
        "dblp")
            # 引用网络
            EPOCHS=100
            LAMBDA=10.0
            BUDGET=0.05
            LR_ATK=0.05
            HIDDEN=64
            ;;
        *)
            echo "Unknown dataset: $DATASET"
            exit 1
            ;;
    esac

    echo -e "\n"
    echo "####################################################################"
    echo ">>> EXPERIMENT: Dataset=[${DATASET}] | Model=[${MODEL}]"
    echo ">>> Config: Epochs=${EPOCHS} | Lambda=${LAMBDA} | Budget=${BUDGET}"
    echo "####################################################################"
    
    # --- 2. 训练攻击 (生成毒药图) ---
    # 输出重定向到日志文件，并在终端显示进度
    LOG_FILE="./logs/${DATASET}_${MODEL}.log"
    echo "Running training... (See $LOG_FILE for details)"
    
    python main.py \
      --dataset ${DATASET} \
      --surrogate_model ${MODEL} \
      --epochs ${EPOCHS} \
      --lambda_fair ${LAMBDA} \
      --ptb_rate ${BUDGET} \
      --lr_atk ${LR_ATK} \
      --hidden_dim ${HIDDEN} \
      --seed 42 \
      --device cuda > ${LOG_FILE} 2>&1
      
    if [ $? -eq 0 ]; then
        echo "Training finished successfully."
    else
        echo "Training FAILED! Check $LOG_FILE"
        return
    fi

    # --- 3. 评估攻击 ---
    echo "Running evaluation..."
    python evaluate_attack.py \
      --dataset ${DATASET} \
      --surrogate_model ${MODEL} \
      --seed 42 \
      --device cuda
      
    echo "--------------------------------------------------------------------"
}

# ==============================================================================
# 主循环
# ==============================================================================

echo "Starting Full Benchmark (6 Datasets x 4 Models)..."

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        run_experiment $DATASET $MODEL
    done
done

echo -e "\nAll 24 experiments completed!"