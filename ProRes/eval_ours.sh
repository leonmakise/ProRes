# !/bin/bash


JOB_NAME="prores_vitl_pretrained_sl1_mprnetprompt_add"
CKPT_DIR="models/${JOB_NAME}"
EPOCH=49
MODEL="uip_vit_large_patch16_input448x448_win_dec64_8glb_sl1"
WORK_DIR="models_inference/${JOB_NAME}"

CUDA_VISIBLE_DEVICES=4 python demo/ours_inference_sidd_v2.py \
  --ckpt_dir ${CKPT_DIR} --model ${MODEL} \
  --epoch ${EPOCH} --input_size 448 --pred_gt --save


# CUDA_VISIBLE_DEVICES=0 python demo/ours_inference_lol_v2.py \
#   --ckpt_dir ${CKPT_DIR} --model ${MODEL} \
#   --epoch ${EPOCH} --input_size 448 --pred_gt --save


# CUDA_VISIBLE_DEVICES=3 python demo/ours_inference_derain_v2.py \
#   --ckpt_dir ${CKPT_DIR} --model ${MODEL}  \
#   --epoch ${EPOCH} --input_size 448 --pred_gt --save --split 3

# CUDA_VISIBLE_DEVICES=3 python demo/ours_inference_deblur_v2.py \
# --ckpt_dir ${CKPT_DIR} --model ${MODEL}  \
# --epoch ${EPOCH} --input_size 448 --pred_gt --save

