#!/usr/bin/env bash
now=$(data +"%Y%m%d_%H%M%S")
EXP_DIR=./sfnets/sfnet_res18_300e_128_inplanes_infer
mkdir -p ${EXP_DIR}
python eval.py \
  --dataset cityscapes \
  --dataset_dir /home/all/datasets/cityscapes \
  --trunk resnet18 \
  --ckpt_path /home/liaoyong/PyCharm/SFSegNet-master/pretrained_models/resnet18-deep-inplane128.pth
  --arch network.sfnet_resnet.DeepR18_SF_deeply \
  --exp cityscapes_SFsegnet_res18 \
