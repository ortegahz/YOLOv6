#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.utils.events import LOGGER


class Model(nn.Module):
    export = False
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, channels=3, num_classes=None, fuse_ab=False, distill_ns=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = 4
        self.backbone, self.neck, self.detect = build_network(channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        x = self.backbone(x)
        x = self.neck(x)
        if not export_mode:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        return x if export_mode is True else [x, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    depth_mul = 0.33
    width_mul = 0.25
    num_repeat_backbone = [1, 6, 12, 18, 6, 6]
    channels_list_backbone = [64, 128, 256, 512, 768, 1024]
    fuse_P2 = True
    cspsppf = True
    num_repeat_neck = [12, 12, 12, 12, 12, 12]
    channels_list_neck = [512, 256, 128, 256, 512, 1024]
    use_dfl = False
    reg_max = 0
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block('repvgg')
    BACKBONE = eval('EfficientRep6')
    NECK = eval('RepBiFPANNeck6')

    backbone = BACKBONE(
        in_channels=channels,
        channels_list=channels_list,
        num_repeats=num_repeat,
        block=block,
        fuse_P2=fuse_P2,
        cspsppf=cspsppf
    )

    neck = NECK(
        channels_list=channels_list,
        num_repeats=num_repeat,
        block=block
    )

    from yolov6.models.effidehead import Detect, build_effidehead_layer
    head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
    head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def build_model(num_classes, device, fuse_ab=False, distill_ns=False):
    model = Model(channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
    return model
