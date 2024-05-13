#!/bin/bash

source activate gqgc

output_dir="./outputs_1"  # Define the output directory

# Create the output directory if it does not exist
mkdir -p $output_dir

# Hyperparameters arrays
declare -a shapes=("16,8,3" "32,8,3" "64,16,3") #"13,13,13" "13,6,3" 
declare -a models=("SAGpool")   #MIAGAE
declare -a depths=(3)
declare -a compression_rates=(0.35 0.36)
declare -a learning_rates=(0.001 0.01)

# Loop through all combinations of hyperparameters
for model in "${models[@]}"
do
    for shape in "${shapes[@]}"
    do
        for depth in "${depths[@]}"
        do
            for c_rate in "${compression_rates[@]}"
            do
                for lr in "${learning_rates[@]}"
                do
                    output_file="${output_dir}/model_${model}_output_shape_${shape}_depth_${depth}_cr_${c_rate}_lr_${lr}.txt"
                    echo "Running model: $model with shape: $shape, depth: $depth, compression rate: $c_rate, learning rate: $lr"
                    # Start the timer
                    start_time=$(date +%s)
                    # Run the python script
                    python -u main.py --shapes $shape --m $model --depth $depth --c_rate $c_rate --lr $lr > "$output_file" 2>&1
                    # Calculate elapsed time
                    end_time=$(date +%s)
                    elapsed_time=$(($end_time - $start_time))
                    elapsed_minutes=$(($elapsed_time / 60))
                    elapsed_seconds=$(($elapsed_time % 60))
                    echo "Elapsed time: ${elapsed_minutes}m ${elapsed_seconds}s" >> "$output_file"
                done
            done
        done
    done
done