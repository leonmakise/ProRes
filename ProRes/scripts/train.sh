#!/bin/bash
# export CC=/cluster_home/custom_data/gcc/bin/gcc
# export CXX=/cluster_home/custom_data/gcc/bin/g++
# export PATH=/cluster_home/custom_data/gcc/bin:$PATH
# export LD_LIBRARY_PATH=/cluster_home/custom_data/gcc/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/cluster_home/custom_data/gcc/lib64:$LD_LIBRARY_PATH


CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

name=ProRes_ep50_lr1e-3
python -m torch.distributed.launch \
  --nnodes=1 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --nproc_per_node=8 \
  --master_port=$PORT \
  --use_env main_pretrain.py  \
  --batch_size 16 \
  --accum_iter 1  \
  --model uip_vit_large_patch16_input448x448_win_dec64_8glb_sl1 \
  --num_mask_patches 784 \
  --max_mask_patches_per_block 0 \
  --epochs 50 \
  --warmup_epochs 2 \
  --lr 1e-3 \
  --weight_decay 0.05  \
  --clip_grad 1.0 \
  --opt_betas 0.9 0.999 \
  --opt_eps 1e-8 \
  --layer_decay 0.8 \
  --drop_path 0.1 \
  --min_random_scale 0.3 \
  --input_size 448 448 \
  --save_freq 1 \
  --data_path ./ \
  --json_path  \
  datasets/low_level/target-derain_train.json \
  datasets/low_level/gt-enhance_lol_train.json \
  datasets/low_level/groundtruth-denoise_ssid_train448.json \
  datasets/low_level/groundtruth_crop-deblur_gopro_train.json \
  --val_json_path \
  datasets/low_level/target-derain_test_rain100h.json \
  datasets/low_level/gt-enhance_lol_eval.json \
  datasets/low_level/groundtruth-denoise_ssid_val256.json \
  datasets/low_level/groundtruth-deblur_gopro_val.json \
  --use_two_pairs \
  --output_dir ./models/$name \
  --log_dir ./models/$name/logs \
  --finetune pretrained_weights/mae_pretrain_vit_large.pth \

#   --log_wandb \
#   --resume models/$name/checkpoint-2.pth \
#   --seed 1000 \
