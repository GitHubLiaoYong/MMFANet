# MMFANet



The implementation model, respectively in the file network.MMFANet3.PSARes18NetV9 and network.MMFANet3.getMMFANet.
The difference between the above two models is which feature fusion method is used in SDM, including addition and channel connection.In our experimental setup, the additive fusion approach yielded better results.

Reproduced Implementation of Our Sensors paper: Multi-Level and Multi-Scale Feature Aggregation Network for
Semantic Segmentation in Vehicle-Mounted Scenes.
 
![avatar](./figs/sfnet_res.png)
Our methods achieve the best speed and accuracy trade-off on Vehicle-Mounted Scenes datasets.  




## DataSet Setting
Please see the DATASETs.md for the details.

## Requirements

pytorch >= 1.2.0
apex
opencv-python



Please download the trained model, the mIoU is on Cityscape validation and test dataset.



The following model is used in the network.MMFANet3.PSARes18NetV9.
The pretrained model used for cityscapes  val dataset: [BaiDuYun](https://pan.baidu.com/s/10WUKCkyVpjIXGJTRuYJLyg ) password: 1234.
The pretrained model used for cityscapes  test dataset: [BaiDuYun](https://pan.baidu.com/s/16isWwb8RpEnibp-s6Fslrw) password: 1234.

## Training 

The train settings require 8 GPU with at least 11GB memory.


Train model
```bash
sh ./scripts/train/train_cityscapes_mmfanet.sh
```





## Acknowledgement 
This repo is based on Semantic Segmentation from [NVIDIA](https://github.com/NVIDIA/semantic-segmentation) and [DecoupleSegNets](https://github.com/lxtGH/DecoupleSegNets) and [SFNet](https://github.com/donnyyou/torchcv)



## Citation
If you find this repo is useful for your research, Please consider citing our paper:


```
Liao, Yong; Liu, Qiong. 2021. "Multi-Level and Multi-Scale Feature Aggregation Network for Semantic Segmentation in Vehicle-Mounted Scenes" Sensors 21, no. 9: 3270. https://doi.org/10.3390/s21093270
```

