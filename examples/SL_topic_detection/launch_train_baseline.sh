#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH --time=24:00:00
#SBATCH --mem 90G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,gpumem:20G


H2S_ROOT=../../../../../../data/How2Sign

FAIRSEQ_ROOT=./

FEATS_TYPE=keypoints
# FEATS_TYPE=mediapipe_keypoints
# FEATS_TYPE=rotational
# FEATS_TYPE=mediapipe_rotational
# FEATS_TYPE=i3d
# FEATS_TYPE=text

SP_MODEL=/mnt/gpid08/users/alvaro.budria/pose2vec/data/How2Sign/text/spm_unigram8000_en.model

# MODEL_TYPE=lstm
# MODEL_TYPE=transformer
MODEL_TYPE=transformerCLS
# MODEL_TYPE=perceiverIO

# CONFIG_NAME=baseline
# CONFIG_NAME=baseline_transformer
# CONFIG_NAME=baseline_perceiverIO
CONFIG_NAME=baseline_${MODEL_TYPE}_${FEATS_TYPE}

NUM_EXP=1

echo $(pwd)

WANDB_MODE=offline SEED=$NUM_EXP fairseq-hydra-train \
    +task.data=${H2S_ROOT}/${FEATS_TYPE} \
    +task.dict_path=${H2S_ROOT}/categoryName_categoryID.csv \
    +task.feats_type=$FEATS_TYPE \
    checkpoint.save_dir=../../../final_models/${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP} \
    bpe.sentencepiece_model=${SP_MODEL} \
    --config-dir ${FAIRSEQ_ROOT}/config \
    --config-name ${CONFIG_NAME}
