#!/usr/bin/env bash
now=$(data +"%Y%m%d_%H%M%S")
EXP_DIR=./sfnets/sfnet_res18_300e_128_inplances_infer
mkdir -p ${EXP_DIR}
CUDA_VISIBLE_DEVICES=6,7,8,9 python eval.py \
  --dataset cityscapes \
  --dataset_dir  /home/all/datasets/cityscapes \
  --trunk resnet-18-deep \
  --ckpt_path /home/liaoyong/PyCharm/SFSegNet-master/eval_result2 \
  --arch network.sfnet_resnet.DeepR18_SF_deeply \
  --exp cityscapes_SFsegNet_res18 \
  --snapshot /home/liaoyong/PyCharm/SFSegNet-master/pretrained_models/res18_sfnet.pth