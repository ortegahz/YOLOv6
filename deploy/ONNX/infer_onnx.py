import argparse
import logging
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run(args):
    img = cv2.imread(args.path_img)

    session = ort.InferenceSession(args.path_weights, providers=['CPUExecutionProvider'])
    names_out = [i.name for i in session.get_outputs()]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    batch = np.expand_dims(img, 0)
    batch = np.ascontiguousarray(batch)
    batch = batch.astype(np.float32) / 255.
    batch = np.ascontiguousarray(batch)

    outputs = session.run(names_out, {'images': batch})
    for i, output in enumerate(outputs):
        np.savetxt(f'/home/manu/tmp/onnx_outputs_{i}.txt', output.flatten(), fmt="%f", delimiter="\n")


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_weights', type=str, default='/home/manu/tmp/acfree.onnx')
    parser.add_argument('--path_img', default='/media/manu/samsung/pics/sylgd_rp.bmp')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
