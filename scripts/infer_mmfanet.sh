#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./sfnets/mmfanet_res18_300e_128_inplances_infer_0
mkdir -p ${EXP_DIR}
CUDA_VISIBLE_DEVICES=1,2,3,4 python eval.py \
  --split test \
  --dataset cityscapes \
  --dataset_dir  /home/all/datasets/cityscapes \
  --trunk resnet-18-deep \
  --ckpt_path /home/liaoyong/PyCharm/SFSegNet-master/eval_result2 \
  --arch network.MMFANet3.PSARes18NetV9 \
  --exp cityscapes_mmfanet_res18_submit_0 \
  --dump_images \
  --snapshot /home/liaoyong/PyCharm/SFSegNet-master/sfets/mmfanet_res18_1000e_128_inplanes/cityscapes_MMFANet_res18_epoch1000_0/city-network.MMFANet3.PSARes18NetV9_apex_T_crop_size_1024_cv_0_lr_0.025_mode_trainva_ohem_T_poly_exp_0.9_restore_optimizer_T_sbn/best_epoch_997_mean-iu_0.88541.pth
# --arch network.sfnet_resnet.DeepR18_SF_deeply \
#-im pooling \
#--single_scale \
#-im whole \
#  --no_flip \