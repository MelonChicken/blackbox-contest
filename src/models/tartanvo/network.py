# BSD 3-Clause attribution:
# Portions are adapted from castacks/tartanvo Network modules.
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu, CMU / Air Lab Stacks.
# See LICENSE for full terms.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .correlation import FunctionCorrelation


def _pwc_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1),
    )


def _predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def _deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class PWCDCNet(nn.Module):
    def __init__(self, md=4, flow_norm=20.0):
        super().__init__()
        self.flow_norm = flow_norm
        self.conv1a = _pwc_conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = _pwc_conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = _pwc_conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = _pwc_conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = _pwc_conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = _pwc_conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = _pwc_conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = _pwc_conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = _pwc_conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = _pwc_conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = _pwc_conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = _pwc_conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = _pwc_conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = _pwc_conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = _pwc_conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = _pwc_conv(128, 196, kernel_size=3, stride=2)
        self.conv6a = _pwc_conv(196, 196, kernel_size=3, stride=1)
        self.conv6b = _pwc_conv(196, 196, kernel_size=3, stride=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        od = nd
        self.conv6_0 = _pwc_conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = _pwc_conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = _pwc_conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = _pwc_conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = _pwc_conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = _predict_flow(od + dd[4])
        self.deconv6 = _deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = _deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        od = nd + 128 + 4
        self.conv5_0 = _pwc_conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = _pwc_conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = _pwc_conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = _pwc_conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = _pwc_conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = _predict_flow(od + dd[4])
        self.deconv5 = _deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = _deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        od = nd + 96 + 4
        self.conv4_0 = _pwc_conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = _pwc_conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = _pwc_conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = _pwc_conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = _pwc_conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = _predict_flow(od + dd[4])
        self.deconv4 = _deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = _deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        od = nd + 64 + 4
        self.conv3_0 = _pwc_conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = _pwc_conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = _pwc_conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = _pwc_conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = _pwc_conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = _predict_flow(od + dd[4])
        self.deconv3 = _deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = _deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        od = nd + 32 + 4
        self.conv2_0 = _pwc_conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = _pwc_conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = _pwc_conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = _pwc_conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = _pwc_conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = _predict_flow(od + dd[4])
        self.deconv2 = _deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.dc_conv1 = _pwc_conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = _pwc_conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = _pwc_conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = _pwc_conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = _pwc_conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = _pwc_conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = _predict_flow(32)

    def warp(self, x, flo):
        b, _, h, w = x.size()
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
        grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).repeat(b, 1, 1, 1)
        vgrid = grid + flo
        vgrid[:, 0] = 2.0 * vgrid[:, 0].clone() / max(w - 1, 1) - 1.0
        vgrid[:, 1] = 2.0 * vgrid[:, 1].clone() / max(h - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        mask = F.grid_sample(torch.ones_like(x), vgrid, align_corners=True)
        mask = (mask >= 0.9999).to(output.dtype)
        return output * mask

    def forward(self, x):
        im1, im2 = x[0], x[1]
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))
        corr6 = self.leakyRELU(FunctionCorrelation(c16, c26))
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)
        corr5 = self.leakyRELU(FunctionCorrelation(c15, self.warp(c25, up_flow6 * 0.625)))
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)
        corr4 = self.leakyRELU(FunctionCorrelation(c14, self.warp(c24, up_flow5 * 1.25)))
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)
        corr3 = self.leakyRELU(FunctionCorrelation(c13, self.warp(c23, up_flow4 * 2.5)))
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)
        corr2 = self.leakyRELU(FunctionCorrelation(c12, self.warp(c22, up_flow3 * 5.0)))
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        return flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))


def _vo_conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=bias), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation), nn.ReLU(inplace=True))


def _linear(in_planes, out_planes):
    return nn.Sequential(nn.Linear(in_planes, out_planes), nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = _vo_conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class VOFlowRes(nn.Module):
    def __init__(self):
        super().__init__()
        inputnum = 4
        blocknums = [2, 2, 3, 4, 6, 7, 3]
        outputnums = [32, 64, 64, 128, 128, 256, 256]
        self.firstconv = nn.Sequential(_vo_conv(inputnum, 32, 3, 2, 1, 1, False), _vo_conv(32, 32, 3, 1, 1, 1), _vo_conv(32, 32, 3, 1, 1, 1))
        self.inplanes = 32
        self.layer1 = self._make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1)
        self.layer5 = self._make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1)
        fcnum = outputnums[6] * 6
        self.voflow_trans = nn.Sequential(_linear(fcnum, 128), _linear(128, 32), nn.Linear(32, 3))
        self.voflow_rot = nn.Sequential(_linear(fcnum, 128), _linear(128, 32), nn.Linear(32, 3))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride)
        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(self.firstconv(x))))))
        x = x.view(x.shape[0], -1)
        return torch.cat((self.voflow_trans(x), self.voflow_rot(x)), dim=1)


class VONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flowNet = PWCDCNet()
        self.flowPoseNet = VOFlowRes()

    def forward(self, x):
        flow = self.flowNet(x[0:2])
        pose = self.flowPoseNet(torch.cat((flow, x[2]), dim=1))
        return flow, pose