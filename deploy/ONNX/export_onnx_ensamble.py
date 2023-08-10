#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import time
from copy import deepcopy

import onnx

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint, save_checkpoint
from yolov6.models.ensamble import EnsambleModel
from io import BytesIO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir_pt', default='/home/manu/tmp/model_phone', type=str)
    parser.add_argument('--export_file_onnx', default='/home/manu/tmp/phone.onnx', type=str)
    parser.add_argument('--weights_player', default='/home/manu/tmp/nn6_ft_b64_nab_s1280_dl/weights/best_ckpt.pt',
                        type=str)
    parser.add_argument('--weights_phone', default='/home/manu/tmp/nn6_ft_b64_nab_s1280_dpc/weights/best_ckpt.pt',
                        type=str)
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640],
                        help='image size, the order is: height width')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--trt-version', type=int, default=8, help='tensorrt version')
    parser.add_argument('--ort', action='store_true', help='export onnx for onnxruntime')
    parser.add_argument('--with-preprocess', action='store_true', help='export bgr2rgb and normalize')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='conf threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model
    model_player = load_checkpoint(args.weights_player, map_location=device, inplace=True, fuse=True)  # load FP32 model
    for layer in model_player.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
            layer.recompute_scale_factor = None  # torch 1.11.0 compatibility
    model_phone = load_checkpoint(args.weights_phone, map_location=device, inplace=True, fuse=True)  # load FP32 model
    for layer in model_player.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
            layer.recompute_scale_factor = None  # torch 1.11.0 compatibility
    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    if args.half:
        img, model_player, model_phone = img.half(), model_player.half(), model_phone.half()  # to FP16
    model_player.eval()
    for k, m in model_player.named_modules():
        if isinstance(m, ConvModule):  # assign export-friendly activations
            if hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace
    model_phone.eval()
    for k, m in model_phone.named_modules():
        if isinstance(m, ConvModule):  # assign export-friendly activations
            if hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace

    model_player.detect.export, model_phone.detect.export_pick = True, True
    model = EnsambleModel(model_player, model_phone)

    y = model(img)  # dry run

    # pytorch export
    ckpt = {
        'model': deepcopy(model).half(),
    }
    save_checkpoint(ckpt, False, args.export_dir_pt, model_name='phone')

    # ONNX export
    try:
        LOGGER.info('\nStarting to export ONNX...')
        export_file = args.export_file_onnx
        with BytesIO() as f:
            torch.onnx.export(model, img, f, verbose=False, opset_version=13,
                              training=torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=True,
                              input_names=['images'],
                              output_names=['outputs'],
                              dynamic_axes=None)
            f.seek(0)
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
        if args.simplify:
            try:
                import onnxsim

                LOGGER.info('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                LOGGER.info(f'Simplifier failure: {e}')
        onnx.save(onnx_model, export_file)
        LOGGER.info(f'ONNX export success, saved as {export_file}')
    except Exception as e:
        LOGGER.info(f'ONNX export failure: {e}')

    # Finish
    LOGGER.info('\nExport complete (%.2fs)' % (time.time() - t))
