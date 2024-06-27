#!/bin/bash

source activate gqgc

output_dir="./outputs_new_vqc"  # Define the output directory

# Create the output directory if it does not exist
mkdir -p $output_dir

# Hyperparameters arrays
#declare -a shapes=("16,8,3" "32,8,3" "64,16,3") #"13,13,13" "13,6,3" 
#declare -a models=("SAGpool")   #MIAGAE
#declare -a depths=(3)
#declare -a compression_rates=(0.35 0.36)
#declare -a learning_rates=(0.001 0.01)


# Classical without compression

declare -a datafolder="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs/"
declare -a train_dataloader_type=("fixed_full" "random_sampling" "fixed_sampling")

#for type in "${train_dataloader_type[@]}"
#do
#    echo "Running classical uncompressed with type: $type"
#    output_file="${output_dir}/VA_classical_uncompressed_${type}.txt"
#    outdir=VA_classical_uncompressed_${type}
#    python -u classical_train.py  --data_folder $datafolder --train_dataloader_type $type --outdir $outdir --input_size 13 > "$output_file" 2>&1

#done

# Classical not guided compression

## Pretrain autoencoders

declare -a models=("SAG_model_vanilla" "MIAGAE_vanilla")

#for aetype in "${models[@]}"
#do
#    echo "Pretraining autoencoder $aetype"
#    dataloader="fixed_full"
#    outdir=VA_pretrain_${aetype}
#    output_file="${output_dir}/VA_pretrain_${aetype}.txt"
#    python -u ae_train.py --data_folder $datafolder --aetype $aetype --epochs 500 --outdir $outdir --train_dataloader_type $dataloader  > "$output_file" 2>&1
#done

## Train classical compressed
#for aetype in "${models[@]}"
#do
#    for type in "${train_dataloader_type[@]}"
#    do
#        echo "Running classical compressed not guided with type: $type and autoencoder $aetype"
#        output_file="${output_dir}/VA_classical_compressed_${aetype}_${type}.txt"
#        outdir=VA_classical_compressed_not_guided_${aetype}_${type}
#        aemodelpath=trained_aes/VA_pretrain_${aetype}/
#        python -u classical_train.py --compressed --aetype $aetype --data_folder $datafolder --ae_model_path $aemodelpath --train_dataloader_type $type --outdir $outdir --input_size 1 > "$output_file" 2>&1
#    done
#done

# Classical guided compression

#declare -a classical_classif_models=("SAG_model_classifier" "MIAGAE_classifier")
#declare -a classical_classif_models=("SAG_model_classifier")

#for classif_model in "${classical_classif_models[@]}"
#do
#    for type in "${train_dataloader_type[@]}"
#    do
#        echo "Running classical guided with type: $type and classifier $classif_model"
#        output_file="${output_dir}/VA_classical_guided_${classif_model}_${type}.txt"
#        outdir=VA_classical_guided_${classif_model}_${type}
#        python -u ae_train.py --data_folder $datafolder --aetype $classif_model --epochs 500 --outdir $outdir --train_dataloader_type $type > "$output_file" 2>&1
#    done
#done


# Quantum not guided compression
#declare -a models=("MIAGAE_vanilla")

#declare -a quantum_train_dataloader_type="fixed_sampling"
#for aetype in "${models[@]}"
#do
#    for type in "${quantum_train_dataloader_type[@]}"
#    do
#        echo "Running quantum compressed not guided with type: $type and autoencoder $aetype"
#        output_file="${output_dir}/VA_quantum_compressed_${aetype}_${type}.txt"
#        outdir=VA_quantum_compressed_not_guided_${aetype}_${type}
#        aemodelpath=trained_aes/VA_pretrain_${aetype}/
#        python -u vqc_train.py --data_folder $datafolder --ae_model_path $aemodelpath --aetype $aetype --train_dataloader_type $type --outdir $outdir --input_size 1 > "$output_file" 2>&1
#    done
#done

# Quantum guided compression
declare -a quantum_classif_models=(SAG_model_vqc_new)
declare -a quantum_train_dataloader_type=("random_sampling" "fixed_sampling" )

for classif_model in "${quantum_classif_models[@]}"
do
    for type in "${quantum_train_dataloader_type[@]}"
    do
        echo "Running quantum guided with type: $type and classifier $classif_model"
        output_file="${output_dir}/VA_quantum_guided_${classif_model}_${type}.txt"
        outdir=VA_quantum_guided_${classif_model}_${type}
        python -u vqc_guided_train.py --data_folder $datafolder --ae_vqc_type $classif_model --train_dataloader_type $type --outdir $outdir > "$output_file" 2>&1
    done
done