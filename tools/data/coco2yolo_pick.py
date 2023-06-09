import argparse
import logging
import shutil
import glob
import cv2
import os

from tqdm import tqdm


def process(args):
    for subset in args.subsets:

        suffix = '2017'
        paths_img = glob.glob(os.path.join(args.dir_root_in, 'images', subset + suffix, '*'))

        for i, path_img in enumerate(tqdm(paths_img)):
            name_img = os.path.basename(path_img)
            name_txt = name_img.replace('.jpg', '.txt')

            path_label = path_img.replace('.jpg', '.txt')
            path_label = path_label.replace('images', 'labels')

            path_subset_txt = os.path.join(args.dir_root_out, subset + '.txt')

            path_img_out = os.path.join(args.dir_root_out, 'images', subset, name_img)
            path_label_out = os.path.join(args.dir_root_out, 'labels', subset, name_txt)

            if not os.path.exists(path_label):
                continue

            with open(path_label, 'r') as f:
                lines = f.readlines()

            with open(path_label_out, 'w') as f:
                cnt = 0
                for line in lines:
                    l, xc, yc, w, h = line.split()
                    if int(l) == args.id_pick:
                        f.writelines(f'0 {xc} {yc} {w} {h}\n')
                        cnt += 1

            if cnt == 0:
                os.remove(path_label_out)
                continue

            shutil.copyfile(path_img, path_img_out)
            with open(path_subset_txt, 'a') as f:
                f.writelines(os.path.join('images', subset, name_img) + '\n')


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def make_dirs(args):
    if os.path.exists(args.dir_root_out):
        shutil.rmtree(args.dir_root_out)
    for subset in args.subsets:
        os.makedirs(os.path.join(args.dir_root_out, 'labels', subset), exist_ok=True)
        os.makedirs(os.path.join(args.dir_root_out, 'images', subset), exist_ok=True)


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root_in', default='/media/manu/kingston/coco', type=str)
    parser.add_argument('--dir_root_out', default='/media/manu/kingstoo/yolov5/coco_phone', type=str)
    parser.add_argument('--subsets', nargs='+', default=['train', 'val'], type=str)
    parser.add_argument('--id_pick', default=67, type=int)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    make_dirs(args)
    process(args)


if __name__ == '__main__':
    main()
