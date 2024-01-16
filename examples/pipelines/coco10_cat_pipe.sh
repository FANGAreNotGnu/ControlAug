#!/bin/sh

source /opt/conda/etc/profile.d/conda.sh

#GPU=1
GPU=$1
#SEED=1
SEED=$2
#SHOT=10
SHOT=$3
#IMSIZE=512
IMSIZE=$4
#NUM_IMG=333
NUM_IMG=$5
#VSP=HED
VSP=$6
#PROMPTMODE=cat
PROMPTMODE=$7
#FILTERRATIO=30
FILTERRATIO=$8
#CKPT_NAME=control_sd15_hed.pth
CKPT_NAME=$9

syn_file="/media/data/ControlAug/cnet/experiments/coco${SHOT}s1_${IMSIZE}p/syn_n${NUM_IMG}_${VSP}_p${IMSIZE}_pr${PROMPTMODE}_dfsNone_seed${SEED}_imprior"
mix_file="/media/data/ControlAug/cnet/experiments/coco${SHOT}s1_${IMSIZE}p/mix_n${NUM_IMG}_${VSP}_p${IMSIZE}_pr${PROMPTMODE}_dfsNone_seed${SEED}_imprior"
filtered_file="/media/data/ControlAug/cnet/experiments/coco${SHOT}s1_${IMSIZE}p/mix_n${NUM_IMG}_${VSP}_p${IMSIZE}_pr${PROMPTMODE}_dfsNone_seed${SEED}_imprior_avgacsl${FILTERRATIO}"


cd "/media/wacv"
cd ControlAug

source scripts/export_paths.sh

conda activate ControlAug_control
CUDA_VISIBLE_DEVICES=${GPU} python3 generate_with_imprior.py \
        -a "/media/data/coco_fsod/seed${SEED}/${SHOT}shot_novel.json" \
        -i /media/data/coco_fsod/train2017/ \
        -e "/media/data/ControlAug/cnet/experiments/coco${SHOT}s1_${IMSIZE}p" \
        -l ${NUM_IMG} \
        -s 1 \
        -p ${IMSIZE} \
        -m 0 \
        --vpg_mode ${VSP} \
        --ckpt_path /media/wacv/ControlNet/models/${CKPT_NAME} \
        --seed ${SEED} \
        --prompt_mode ${PROMPTMODE}
        
python3 mix_annotations.py \
        -a "/media/data/coco_fsod/seed${SEED}/${SHOT}shot_novel.json" \
        -s ${syn_file} \
        -t ${mix_file} \
        -f nofilter \
        -n ${NUM_IMG}
conda deactivate

conda activate ControlAug_clip
CUDA_VISIBLE_DEVICES=${GPU} python3 calculate_clip_score.py \
        -d ${mix_file}

python3 filter_annotations.py \
        -d ${mix_file} \
        -k csl \
        -p ${FILTERRATIO}
conda deactivate

echo ${filtered_file}
