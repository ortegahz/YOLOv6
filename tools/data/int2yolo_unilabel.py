import argparse
import logging
import shutil
import glob
import cv2
import os

from tqdm import tqdm


def process(args):
    logging.info(args)

    paths_img = glob.glob(os.path.join(args.dir_imgs_in, '*'))

    for i, path_img in enumerate(tqdm(paths_img)):
        name_img = os.path.basename(path_img)
        name_txt = name_img.replace('.jpg', '.txt')

        path_label = path_img.replace('.jpg', '.txt')
        path_label = path_label.replace('imgs', 'labels')

        path_trainset_txt = os.path.join(args.dir_root_out, 'train.txt')
        path_valset_txt = os.path.join(args.dir_root_out, 'val.txt')

        with open(path_trainset_txt, 'a') as ft, open(path_valset_txt, 'a') as fv:
            if i < args.rate_val * len(paths_img):
                path_img_out = os.path.join(args.dir_root_out, 'images', 'val', name_img)
                path_label_out = os.path.join(args.dir_root_out, 'labels', 'val', name_txt)
                fv.writelines(os.path.join('images', 'val', name_img) + '\n')
            else:
                path_img_out = os.path.join(args.dir_root_out, 'images', 'train', name_img)
                path_label_out = os.path.join(args.dir_root_out, 'labels', 'train', name_txt)
                ft.writelines(os.path.join('images', 'train', name_img) + '\n')

        shutil.copyfile(path_img, path_img_out)
        if os.path.exists(path_label):
            img = cv2.imread(path_img)
            h_img, w_img, _ = img.shape

            with open(path_label, 'r') as f:
                lines = f.readlines()

            with open(path_label_out, 'w') as f:
                for line in lines:
                    l, xc, yc, w, h = line.split()
                    f.writelines(f'0 {xc} {yc} {w} {h}\n')



def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def make_dirs(args):
    if os.path.exists(args.dir_root_out):
        shutil.rmtree(args.dir_root_out)
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(args.dir_root_out, 'labels', subset), exist_ok=True)
        os.makedirs(os.path.join(args.dir_root_out, 'images', subset), exist_ok=True)


def parse_ars():
    set_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_imgs_in', default='/media/manu/kingstoo/yolov5/custom_head_v1/merge/imgs', type=str)
    parser.add_argument('--dir_root_out', default='/media/manu/kingstoo/yolov5/custom_head_v1_f', type=str)
    parser.add_argument('--rate_val', default=5000 / 118287., type=float)
    return parser.parse_args()


def main():
    args = parse_ars()
    make_dirs(args)
    process(args)


if __name__ == '__main__':
    main()
