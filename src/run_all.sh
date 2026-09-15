#!/bin/bash

# Define the full 12-activation study
experiments=(
    # Piecewise-linear / classical
    "relu Exp1_ReLU"
    "leaky_relu Exp2_Leaky"
    "prelu Exp3_PReLU"
    "elu Exp4_ELU"

    # Smooth, self-gated
    "gelu Exp5_GELU"
    "swish Exp6_Swish"
    "mish Exp7_Mish"

    # Recent smooth variants
    "elish Exp8_ELiSH"
    "hard_elish Exp9_HardELiSH"
    "logish Exp10_Logish"
    "smish Exp11_Smish"
    "tanhexp Exp12_TanhExp"
)

echo "Launching the 12-activation study..."
echo "=================================================="

for exp in "${experiments[@]}"; do
    set -- $exp
    ACT=$1
    NAME=$2

    echo "Starting: $NAME (activation: $ACT)"

    # Each run resumes automatically from latest.pth if interrupted
    python train.py --act $ACT --name $NAME --epochs 100

    echo "Finished: $NAME"
    echo "--------------------------------------"
done

echo "All experiments complete."
