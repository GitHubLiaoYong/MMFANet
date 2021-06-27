#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./sfets/mmfanet_res18_1000e_128_inplanes
mkdir -p ${EXP_DIR}
# Example on Cityscapes by resnet50-deeplabv3+ as baseline
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=6 --master_port 4932 train.py \
  --dataset cityscapes \
  --mode train \
  --cv 0 \
  --arch network.MMFANet3.Paper_MMFANet2 \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --lr 0.025 \
  --lr_schedule poly \
  --poly_exp 0.9 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --ohem \
  --crop_size 1024 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --max_epoch 1000 \
  --wt_bound 1.0 \
  --bs_mult 2 \
  --apex \
  --exp cityscapes_MMFANet_res18_epoch1000 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &
#--snapshot /home/liaoyong/PyCharm/SFSegNet-master/pretrained_models/resnet18-deep-inplane128.pth \
#--snapshot /home/liaoyong/PyCharm/SFSegNet-master/sfets/sfnet_res18_300e_128_inplanes/cityscapes_MMFANet_res18_1/city-network.MMFANet3.PSARes18NetV9_apex_T_bs_mult_4_crop_size_1024_cv_0_lr_0.01_ohem_T_sbn/last_epoch_423_mean-iu_0.73525.pth \
#--restore_optimizer \
