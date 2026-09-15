#!/bin/bash

# Define the full 12-activation study
experiments=(
    # Group 1: Baselines
    "relu Exp1_ReLU"
    "leaky_relu Exp2_Leaky"
    "prelu Exp3_PReLU"
    "elu Exp4_ELU"

    # Group 2: Modern
    "gelu Exp5_GELU"
    "swish Exp6_Swish"
    "mish Exp7_Mish"

    # Group 3: Advanced/Novel
    "elish Exp8_ELiSH"
    "hard_elish Exp9_HardELiSH"
    "logish Exp10_Logish"
    "smish Exp11_Smish"
    "tanhexp Exp12_TanhExp"
)

echo "🚀 LAUNCHING COMPREHENSIVE 12-ACTIVATION STUDY..."
echo "=================================================="

for exp in "${experiments[@]}"; do
    set -- $exp
    ACT=$1
    NAME=$2

    echo "▶️  STARTING: $NAME (Act: $ACT)"

    # Run Script (Auto-resumes if interrupted)
    python train.py --act $ACT --name $NAME --epochs 100

    echo "✅ FINISHED: $NAME"
    echo "--------------------------------------"
done

echo "🎉 ALL EXPERIMENTS COMPLETE!"
