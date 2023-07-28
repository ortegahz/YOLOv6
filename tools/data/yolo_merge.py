import argparse
import logging
import shutil
import glob
import os
import cv2

from tqdm import tqdm


def process(args):
    for dir_root in args.dir_roots_in:
        for subset in args.subsets:

            paths_img = glob.glob(os.path.join(dir_root, 'images', subset, '*'))

            for path_img in tqdm(paths_img):
                name_img = os.path.basename(path_img)
                name_txt = name_img.replace('.jpg', '.txt')

                path_label = path_img.replace('.jpg', '.txt')
                path_label = path_label.replace('images', 'labels')

                path_subset_txt = os.path.join(args.dir_root_out, subset + '.txt')

                path_img_out = os.path.join(args.dir_root_out, 'images', subset, name_img)
                path_label_out = os.path.join(args.dir_root_out, 'labels', subset, name_txt)

                if not os.path.exists(path_label):
                    continue

                shutil.copyfile(path_img, path_img_out)
                shutil.copyfile(path_label, path_label_out)
                with open(path_subset_txt, 'a') as f:
                    f.writelines(os.path.join('images', subset, name_img) + '\n')


def make_dirs(args):
    if os.path.exists(args.dir_root_out):
        shutil.rmtree(args.dir_root_out)
    for subset in args.subsets:
        os.makedirs(os.path.join(args.dir_root_out, 'labels', subset), exist_ok=True)
        os.makedirs(os.path.join(args.dir_root_out, 'images', subset), exist_ok=True)


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_roots_in', nargs='+',
                        default=['/media/sdb/data/custom_head_v1r',
                                 '/media/sdb/data/custom_head_v2_f'],
                        type=str)
    parser.add_argument('--dir_root_out', default='/media/sdb/data/custom_head_v2r', type=str)
    parser.add_argument('--subsets', nargs='+', default=['train', 'val'], type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    make_dirs(args)
    process(args)


if __name__ == '__main__':
    main()
