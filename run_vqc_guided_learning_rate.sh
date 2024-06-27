#!/bin/bash

source activate gqgc

output_dir="./outputs_vqc_guided_learning_rate"  # Define the output directory

# Create the output directory if it does not exist
mkdir -p $output_dir

# Hyperparameters arrays
#declare -a shapes=("16,8,3" "32,8,3" "64,16,3") #"13,13,13" "13,6,3" 
#declare -a models=("SAGpool")   #MIAGAE
#declare -a depths=(3)
#declare -a compression_rates=(0.35 0.36)
#declare -a learning_rates=(0.001 0.01)

declare -a n_layers=(2 3 4)
declare -a models=("SAG_model_vqc" "MIAGAE_vqc")
declare -a datafolder=("data/graphdata_10000_part_dist_std" "data/graphdata_10000_part_dist_maxabs" "data/graphdata_10000_part_dist_minmax")


for n_layer in "${n_layers[@]}"
do
    for model in "${models[@]}"
    do
        for data in "${datafolder[@]}"
        do
                    output_file="${output_dir}/model_${model}_nlayers_${n_layer}_data_$(basename ${data}).txt"
                    echo "Running model: $model with nlayers: $n_layer, data: $data"
                    # Start the timer
                    start_time=$(date +%s)
                    outdir=model_${model}_nlayers_${n_layer}_data_$(basename ${data})
                    # Run the python script
                    python -u vqc_guided_train.py --n_layers $n_layer --ae_vqc_type $model --data_folder $data --outdir "$outdir" --lr 0.001 > "$output_file" 2>&1
                    # Calculate elapsed time
                    end_time=$(date +%s)
                    elapsed_time=$(($end_time - $start_time))
                    elapsed_minutes=$(($elapsed_time / 60))
                    elapsed_seconds=$(($elapsed_time % 60))
                    echo "Elapsed time: ${elapsed_minutes}m ${elapsed_seconds}s" #>> "$output_file"
        done
    done
done