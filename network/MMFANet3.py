import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from network.nn.mynn import Norm2d, Upsample
from network.resnet_d import BasicBlock
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False,
                 dilation=1,groups=1,stride=1,normal_layer=nn.BatchNorm2d):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', normal_layer(num_maps_in))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias,
                                          dilation=dilation,groups=groups,stride=stride))
class deepwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 BatchNorm=nn.BatchNorm2d,bias=False):
        super().__init__()
        self.depthConv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=in_channels,bias=bias)
        self.pointwiseConv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,bias=bias)
        # self.bn = BatchNorm(out_channels)
        # self.relu = nn.ReLU(out_channels)

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointwiseConv(x)
        # x = self.bn(x)
        # x = self.relu(x)
        return x
class _BNReluConvDeepWise(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False,
                 dilation=1,groups=1,stride=1,normal_layer=nn.BatchNorm2d):
        super(_BNReluConvDeepWise, self).__init__()
        if batch_norm:
            self.add_module('norm', normal_layer(num_maps_in))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', deepwiseConv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding,
                                          dilation=dilation,groups=groups,stride=stride,BatchNorm=normal_layer))

class HighFFM(nn.Module):
    """

    """
    def __init__(self,in_channels,out_channels=256,normal_layer=nn.BatchNorm2d):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
        self.psa = Bottle2neckMulDilatedHalf2(inplanes=out_channels,planes=out_channels,normal_layer=normal_layer)
        self.conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1)
        self.bn = normal_layer(out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    def forward(self, low_in,high_in):
        high_in = F.interpolate(high_in,low_in.size()[2:],mode='bilinear',align_corners=False)#上采样
        x = torch.cat((low_in,high_in),dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.psa(x)
        p = self.avgpool(x)
        p = self.conv(p)
        p = self.sigmoid(p)
        x = torch.mul(x,p)
        return x


class spatial_fuse(nn.Module):
    """
    Use add
    """
    def __init__(self, low_solution_channels, high_solution_channels,use_bn=True,k=3,normal_layer=nn.BatchNorm2d):
        super(spatial_fuse, self).__init__()
        self.low_solution_channels = low_solution_channels
        self.high_solution_channels = high_solution_channels
        self.bottleneck = _BNReluConv(num_maps_in=high_solution_channels,num_maps_out=low_solution_channels,k=1,batch_norm=use_bn,stride=2,normal_layer=normal_layer)
        self.blend_conv = _BNReluConv(num_maps_in=low_solution_channels,num_maps_out=high_solution_channels,k=k,batch_norm=use_bn,normal_layer=normal_layer)#groups=high_solution_channels)#这里可以设置上采样中卷积是否使用深度可分

    def forward(self, x):
        up_size = x[1].size()[2:]
        skip = self.bottleneck.forward(x[1])
        x = skip + x[0]
        x = self.blend_conv.forward(x)#
        return x
class SDM(nn.Module):
    """
    Use concatenate
    """
    def __init__(self, low_solution_channels=128, high_solution_channels=64,use_bn=True,k=3,normal_layer=nn.BatchNorm2d):
        super(SDM, self).__init__()
        self.low_solution_channels = low_solution_channels
        self.high_solution_channels = high_solution_channels
        self.bottleneck = _BNReluConv(num_maps_in=high_solution_channels,num_maps_out=low_solution_channels,k=1,batch_norm=use_bn,stride=2,normal_layer=normal_layer)#64-》128
        self.blend_conv = _BNReluConv(num_maps_in=low_solution_channels * 2,num_maps_out=high_solution_channels,k=k,batch_norm=use_bn,normal_layer=normal_layer)#256-》64 groups=high_solution_channels)#这里可以设置上采样中卷积是否使用深度可分


    def forward(self, x):
        up_size = x[1].size()[2:]
        skip = self.bottleneck.forward(x[1])
        x = torch.cat((skip,x[0]),1)
        x = self.blend_conv.forward(x)#
        return x

class SDM2(nn.Module):
    """
    Use concatenate
    """
    def __init__(self, low_solution_channels=128, high_solution_channels=64,use_bn=True,k=3,normal_layer=nn.BatchNorm2d):
        super(SDM2, self).__init__()
        self.low_solution_channels = low_solution_channels
        self.high_solution_channels = high_solution_channels
        self.bottleneck = _BNReluConv(num_maps_in=high_solution_channels,num_maps_out=high_solution_channels,k=1,batch_norm=use_bn,stride=2,normal_layer=normal_layer)#64-》128
        self.blend_conv = _BNReluConv(num_maps_in=low_solution_channels + high_solution_channels,num_maps_out=high_solution_channels,k=k,batch_norm=use_bn,normal_layer=normal_layer)

    def forward(self, x):
        up_size = x[1].size()[2:]
        skip = self.bottleneck.forward(x[1])
        x = torch.cat((skip,x[0]),1)
        x = self.blend_conv.forward(x)#
        return x

class Decoder2(nn.Module):
    def __init__(self,num_classes,normal_layer=nn.BatchNorm2d,class_in_planes=304,low_level_inplances=32,mid_channels=256):
        super(Decoder2,self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplances,48,1,bias=False)
        self.bn1 = normal_layer(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(class_in_planes,mid_channels,kernel_size=3,stride=1,padding=1,bias=False),
                                       normal_layer(mid_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(mid_channels,mid_channels,kernel_size=3,stride=1,padding=1,bias=False),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(mid_channels,num_classes,kernel_size=1,stride=1))
        self._init_weight()

    def forward(self, x,low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x,size=low_level_feat.size()[2:],mode='bilinear',align_corners=True)

        x = torch.cat((x,low_level_feat),dim=1)
        x = self.last_conv(x)

        return x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self,num_classes,normal_layer=nn.BatchNorm2d,class_in_planes=304,low_level_inplances=32,mid_channels=256):
        super(Decoder,self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplances,48,1,bias=False)
        self.bn1 = normal_layer(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(class_in_planes,mid_channels,kernel_size=3,stride=1,padding=1,bias=False),
                                       normal_layer(mid_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(mid_channels,mid_channels,kernel_size=3,stride=1,padding=1,bias=False),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(mid_channels,num_classes,kernel_size=1,stride=1))
        self._init_weight()

    def forward(self, x,low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x,size=low_level_feat.size()[2:],mode='bilinear',align_corners=True)
        x = torch.cat((x,low_level_feat),dim=1)
        x = self.last_conv(x)
        return x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
class Bottle2neckMulDilatedHalf(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilated=1, downsample=None, baseWidth=24, scale=4, stype='normal',
                 BatchNorm=Norm2d):
        """ Constructor
        有多个空洞率的并行卷积
        1 2 5 9；5 9 17
        让标准卷积占用1/2的通道数

        将标准卷积分支设置的通道数占据baseWidth*2
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.如果设置了步长可以替换池化层
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neckMulDilatedHalf, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.stdC = 2 * width
        self.dilatedC = math.floor((2*width)/3)
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        dil = [1,2,5,9]
        dil2 = [1,5,9,17]
        for i in range(self.nums):
            if i == 0:
                convs.append(
                    nn.Conv2d(self.stdC, self.stdC, kernel_size=3, stride=stride, dilation=2 ** i, padding=2 ** i, bias=False)
                )
                bns.append(BatchNorm(width*2))
            else:
                convs.append(
                    nn.Conv2d(self.dilatedC, self.dilatedC, kernel_size=3, stride=stride, dilation=dil[i], padding=dil[i], bias=False))
                bns.append(BatchNorm(self.dilatedC))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width * 2, 1)
        spy = torch.split(spx[1],self.dilatedC,1)
        for i in range(self.nums):
            if i == 0 :
                sp = spx[0]
            elif i == 1:
                tsp = torch.split(sp,self.dilatedC,1)
                s1 = tsp[0].size()[2]
                s2 = spy[i-1].size()[2]
                if(s1 != s2):
                    tsp0 = F.interpolate(tsp[0],spy[i-1].size()[2:],mode='bilinear',align_corners=False)
                    tsp1 = F.interpolate(tsp[1], spy[i-1].size()[2:], mode='bilinear', align_corners=False)
                    tsp2 = F.interpolate(tsp[2], spy[i-1].size()[2:], mode='bilinear', align_corners=False)
                    sp = tsp0 + tsp1 + tsp2 + spy[i-1]
                else:
                    sp = tsp[0] + tsp[1] + tsp[2] + spy[i-1]
            else:
                s1 = sp.size()[2]
                s2 = spy[i-1].size()[2]
                if s1 != s2:
                    sp = F.interpolate(sp,spy[i-1].size()[2:],mode='bilinear',align_corners=False)
                sp = sp + spy[i-1]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class splitConv(nn.Module):
    def __init__(self,nIn,nOut,dkSize,d,stride,padding,dilation=(1,1),groups=1,bn_acti=False):
        super().__init__()

        self.bn_acti = bn_acti

        '''
        输入输出通道数一定相同
        '''
        self.ddconv3x1 = Conv(nIn , nIn , (dkSize, 1),stride=(stride,1),
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn , bn_acti=True)
        self.ddconv1x3 = Conv(nIn , nIn , (1, dkSize),stride=(1,stride),
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn, bn_acti=True)

    def forward(self,input):
            output = self.ddconv3x1(input)
            output = self.ddconv1x3(output)

            return output


class PSAResNetV9(nn.Module):
    """
    This model is MMFANet, but in SDM we use element-wise add to fuse features. This model can get the best results.
    """
    def __init__(self, block, psa_block, layers, classes=19, baseWidth=24, scale=4, num_classes=1000,
                 use_bn=True, efficient=False, separable=False, mean=(73.1584, 82.9090, 72.3924),
                 std=(44.9149, 46.1529, 45.3192),normal_layer=nn.BatchNorm2d,criterion=None):

        super(PSAResNetV9, self).__init__()
        self.classes = classes
        self.criterion = criterion
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.separable = separable
        self.efficient = efficient
        self.scale = scale
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = normal_layer(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_res_layer(block, 64, layers[0])

        self.layer2 = self._make_res_layer(block, 128, layers[1], stride=2)

        self.psa3 = self._make_layer(psa_block, 256, layers[2], stride=2)
        self.psa4 = self._make_layer(psa_block, 256, layers[3], stride=2)
        self.fusion = HighFFM(in_channels=512,out_channels=256,normal_layer=normal_layer)
        self.low_laver = spatial_fuse(low_solution_channels=128, high_solution_channels=64,
                                      normal_layer=normal_layer)
        self.decoder = Decoder(num_classes=classes,normal_layer =normal_layer, low_level_inplances=64,
        class_in_planes = 304)
        self.init_layers = [self.fusion, self.decoder, self.psa4, self.psa3, self.low_laver]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_1x_lr_params(self):
        if self.classes == 11:
            modules = [self.layer1,self.layer2,self.conv1,self.bn1,self.psa3,self.psa4,self.low_laver,self.fusion]
        else:
            modules = [self.layer2, self.layer1, self.conv1, self.bn1]

        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        if self.classes == 11:
            modules = [self.decoder]
        else:
            modules = [self.fusion, self.decoder, self.psa4, self.psa3, self.low_laver]
        #self.layer4_class,self.layer3_class,
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image,gts=None):
        out_size = image.size()[2:]
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        size1 = x.size()[2:]

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [x]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [x]
        x = self.psa3(x)  # 256, 1/16
        features += [x]
        x = self.psa4(x)  # 256 1/32
        features += [x]
        out = self.fusion(features[2], features[3])
        low_level = self.low_laver((features[1], features[0]))
        out = self.decoder(out, low_level)
        main_out = F.interpolate(out, out_size, mode='bilinear', align_corners=False)
        if self.training:
            return self.criterion(main_out,gts)

        return main_out

    def forward(self, input,gts=None):
        return self.forward_down(input,gts)

    def _make_res_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [Norm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes)]

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

class MMFANet(nn.Module):
    """
    This model also is MMFANet, but in SDM we use concatenate to fuse features.
    """

    def __init__(self, block, psa_block, layers, classes=19, baseWidth=24, scale=4, num_classes=1000,
                 use_bn=True, efficient=False, separable=False, mean=(73.1584, 82.9090, 72.3924),
                 std=(44.9149, 46.1529, 45.3192),normal_layer=nn.BatchNorm2d,criterion=None):

        super(MMFANet, self).__init__()
        self.classes = classes
        self.criterion = criterion
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.separable = separable
        self.efficient = efficient
        self.scale = scale
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = normal_layer(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_res_layer(block, 64, layers[0])

        self.layer2 = self._make_res_layer(block, 128, layers[1], stride=2)

        self.psa3 = self._make_layer(psa_block, 256, layers[2], stride=2)
        self.psa4 = self._make_layer(psa_block, 256, layers[3], stride=2)
        self.fusion = HighFFM(in_channels=512,out_channels=256,normal_layer=normal_layer)
        self.low_laver = SDM2(low_solution_channels=128, high_solution_channels=64,
                                      normal_layer=normal_layer)
        self.decoder = Decoder(num_classes=classes,normal_layer =normal_layer, low_level_inplances=64,
        class_in_planes = 304)
        self.init_layers = [self.fusion, self.decoder, self.psa4, self.psa3, self.low_laver]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_1x_lr_params(self):
        if self.classes == 11:
            modules = [self.layer1,self.layer2,self.conv1,self.bn1,self.psa3,self.psa4,self.low_laver,self.fusion]
        else:
            modules = [self.layer2, self.layer1, self.conv1, self.bn1]

        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        if self.classes == 11:
            modules = [self.decoder]
        else:
            modules = [self.fusion, self.decoder, self.psa4, self.psa3, self.low_laver]
        #self.layer4_class,self.layer3_class,
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image,gts=None):
        out_size = image.size()[2:]
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        size1 = x.size()[2:]

        features = []
        x, skip = self.forward_resblock(x, self.layer1)  # 64 1/4
        features += [x]
        x, skip = self.forward_resblock(x, self.layer2)  # 128 1/8
        features += [x]
        x = self.psa3(x)  # 256, 1/16
        features += [x]
        x = self.psa4(x)  # 256 1/32
        features += [x]

        out = self.fusion(features[2], features[3])#out 1/16 256
        low_level = self.low_laver((features[1], features[0]))#low_level 64 1/8
        out = self.decoder(out, low_level)
        main_out = F.interpolate(out, out_size, mode='bilinear', align_corners=False)

        if self.training:
            return self.criterion(main_out,gts)

        return main_out

    def forward(self, input,gts=None):
        return self.forward_down(input,gts)

    def _make_res_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [Norm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes)]

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

def PSARes18NetV9(pretrained=True,num_classes=19,criterion=None):
    model = PSAResNetV9(block=BasicBlock, psa_block=Bottle2neckMulDilatedHalf, layers=[2, 2, 2, 2],
                        baseWidth=24, scale=4,classes=num_classes,criterion=criterion,normal_layer=Norm2d,num_classes=num_classes)#classes=11 camvid
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def getMMFANet(pretrained=True,num_classes=19,criterion=None):
    model = MMFANet(block=BasicBlock, psa_block=Bottle2neckMulDilatedHalf, layers=[2, 2, 2, 2],
                        baseWidth=24, scale=4,classes=num_classes,criterion=criterion,normal_layer=Norm2d,num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

    return model