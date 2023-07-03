import argparse
import glob
import logging
import os
import shutil

from tqdm import tqdm


def run(args):
    for idx, dir_root_in in enumerate(args.dirs_root_in):
        paths_img = glob.glob(os.path.join(dir_root_in, 'imgs', '*'))

        for path_img in tqdm(paths_img):
            path_label = path_img.replace('.jpg', '.txt')
            path_label = path_label.replace('imgs', 'labels')

            path_img_out = path_img.replace(dir_root_in, args.dir_root_out)
            path_label_out = path_label.replace(dir_root_in, args.dir_root_out)

            path_img_out = path_img_out[:-4] + f'_{idx}' + path_img_out[-4:]
            path_label_out = path_label_out[:-4] + f'_{idx}' + path_label_out[-4:]

            assert not os.path.exists(path_img_out) and not os.path.exists(path_label_out),\
                f'{path_img_out} \n {path_label_out} \n'

            shutil.copyfile(path_img, path_img_out)
            if os.path.exists(path_label):
                shutil.copyfile(path_label, path_label_out)


def make_dirs(args):
    if os.path.exists(args.dir_root_out):
        shutil.rmtree(args.dir_root_out)
    os.makedirs(os.path.join(args.dir_root_out, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(args.dir_root_out, 'imgs'), exist_ok=True)


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_ars():
    set_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs_root_in', nargs='+',
                        default=['/media/manu/kingstoo/yolov5/custom_head_v1/r1',
                                 '/media/manu/kingstoo/yolov5/custom_head_v1/r2'],
                        type=str)
    parser.add_argument('--dir_root_out', default='/media/manu/kingstoo/yolov5/custom_head_v1/merge', type=str)
    return parser.parse_args()


def main():
    args = parse_ars()
    logging.info(args)
    make_dirs(args)
    run(args)


if __name__ == '__main__':
    main()
