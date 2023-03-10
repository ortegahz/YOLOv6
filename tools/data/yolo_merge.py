import logging
import argparse
import shutil
import os

from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('manu')


def process(args):
    logger.info(args)
    for dir_in_root in args.dirs_in_root:
        for subset in args.subsets:
            path_subset_txt_i = os.path.join(dir_in_root, f'{subset}2017.txt')
            path_subset_txt_o = os.path.join(args.dir_out_root, f'{subset}2017.txt')
            with open(path_subset_txt_i, 'r') as fsti:
                lines_i = fsti.readlines()
            with open(path_subset_txt_o, 'a') as fsto:
                for idx, line_i in enumerate(tqdm(lines_i)):
                    fsto.writelines(line_i)
                    path_img_i = os.path.join(dir_in_root, line_i)[:-1]
                    path_img_o = os.path.join(args.dir_out_root, line_i)[:-1]
                    shutil.copyfile(path_img_i, path_img_o)
                    path_lb_i = path_img_i.replace('images', 'labels')
                    path_lb_i = path_lb_i.replace('.jpg', '.txt')
                    path_lb_o = path_img_o.replace('images', 'labels')
                    path_lb_o = path_lb_o.replace('.jpg', '.txt')
                    shutil.copyfile(path_lb_i, path_lb_o)


def new_dirs(args):
    if os.path.exists(args.dir_out_root):
        shutil.rmtree(args.dir_out_root)
    os.makedirs(args.dir_out_root)
    dir_out_imgs = os.path.join(args.dir_out_root, 'images')
    os.makedirs(dir_out_imgs)
    for subset in args.subsets:
        dir_out_imgs_subset = os.path.join(dir_out_imgs, f'{subset}2017')
        os.makedirs(dir_out_imgs_subset)
    dir_out_lbs = os.path.join(args.dir_out_root, 'labels')
    os.makedirs(dir_out_lbs)
    for subset in args.subsets:
        dir_out_lbs_subset = os.path.join(dir_out_lbs, f'{subset}2017')
        os.makedirs(dir_out_lbs_subset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs_in_root', nargs='*', default=['/media/manu/kingstoo/yolov5/custom_widerface',
                                                              '/media/manu/kingstoo/yolov5/custom_head'], type=str)
    parser.add_argument('--dir_out_root', default='/media/manu/kingstoo/yolov5/custom_acfree', type=str)
    parser.add_argument('--subsets', nargs='*', default=['train', 'val'], type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    new_dirs(args)
    process(args)


if __name__ == '__main__':
    main()