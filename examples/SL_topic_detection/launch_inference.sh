#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH --time=24:00:00
#SBATCH --mem 40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,gpumem:12G


# FEATS_TYPE=keypoints
# FEATS_TYPE=mediapipe_keypoints
# FEATS_TYPE=rotational
# FEATS_TYPE=mediapipe_rotational
# FEATS_TYPE=i3d
# FEATS_TYPE=text
# FEATS_TYPE=spot_align
FEATS_TYPE=spot_align_pretrained_text

H2S_ROOT=../../../../../../data/How2Sign/${FEATS_TYPE}

FAIRSEQ_ROOT=../../..

# MODEL_TYPE=lstm
# MODEL_TYPE=transformer
# MODEL_TYPE=transformerCLS
MODEL_TYPE=perceiverIO

CONFIG_NAME=inference_${MODEL_TYPE}

# SP_MODEL=/home/alvaro/Documents/ML_and_DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/text/spm_unigram251_en.model
# SP_MODEL=${H2S_ROOT}/${SP_MODEL} \
SP_MODEL=/mnt/gpid08/users/alvaro.budria/pose2vec/data/How2Sign/text/spm_unigram8000_en.model

OUTPUTS_DIR=${FAIRSEQ_ROOT}/outputs
mkdir -p $OUTPUTS_DIR

echo $(pwd)

for NUM_EXP in 1 2 3
do
    echo NUM_EXP = ${NUM_EXP}
    if [ "$FEATS_TYPE" = "spot_align_pretrained_text" ]; then
        MODEL_PATH=../../../final_models/${MODEL_TYPE}_text_${NUM_EXP}/checkpoint_best.pt
        H2S_ROOT=../../../../../../data/How2Sign/spot_align
        for DATASET_SPLIT in val test train
        do
            echo '*************************************************'
            echo Starting experiment $NUM_EXP, $DATASET_SPLIT split, $FEATS_TYPE features
            echo '*************************************************'
            DATA=$H2S_ROOT \
            DICT_PATH=${H2S_ROOT}/categoryName_categoryID.csv \
            MODEL_PATH=$MODEL_PATH \
            CONFIG_NAME=${CONFIG_NAME} \
            SP_MODEL=${SP_MODEL} \
            DATASET_SPLIT=$DATASET_SPLIT \
            OUTPUTS_FILE=${OUTPUTS_DIR}/${CONFIG_NAME}_${FEATS_TYPE}_${DATASET_SPLIT}.pt \
            FEATS_TYPE=spot_align \
            python infer.py
            echo '*************************************************'
            echo Finishing experiment $NUM_EXP, $DATASET_SPLIT split
            echo '*************************************************'
            echo
        done
    else
        MODEL_PATH=../../../final_models/${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP}/checkpoint_best.pt
        for DATASET_SPLIT in val test train
        do
            echo '*************************************************'
            echo Starting experiment $NUM_EXP, $DATASET_SPLIT split, $FEATS_TYPE features
            echo '*************************************************'
            DATA=$H2S_ROOT \
            DICT_PATH=${H2S_ROOT}/categoryName_categoryID.csv \
            MODEL_PATH=$MODEL_PATH \
            CONFIG_NAME=${CONFIG_NAME} \
            SP_MODEL=${SP_MODEL} \
            DATASET_SPLIT=$DATASET_SPLIT \
            OUTPUTS_FILE=${OUTPUTS_DIR}/${CONFIG_NAME}_${FEATS_TYPE}_${DATASET_SPLIT}.pt \
            FEATS_TYPE=$FEATS_TYPE \
            python infer.py
            echo '*************************************************'
            echo Finishing experiment $NUM_EXP, $DATASET_SPLIT split
            echo '*************************************************'
            echo
        done
    fi
done
