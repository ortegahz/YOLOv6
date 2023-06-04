import argparse
import logging
import shutil
import glob
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

        path_trainset_txt = os.path.join(args.dir_root_out, 'train2017.txt')
        path_valset_txt = os.path.join(args.dir_root_out, 'val2017.txt')

        with open(path_trainset_txt, 'a') as ft, open(path_valset_txt, 'a') as fv:
            if i < args.rate_val * len(paths_img):
                path_img_out = os.path.join(args.dir_root_out, 'images', 'val' + '2017', name_img)
                path_label_out = os.path.join(args.dir_root_out, 'labels', 'val' + '2017', name_txt)
                fv.writelines(os.path.join('images', 'val' + '2017', name_img) + '\n')
            else:
                path_img_out = os.path.join(args.dir_root_out, 'images', 'train' + '2017', name_img)
                path_label_out = os.path.join(args.dir_root_out, 'labels', 'train' + '2017', name_txt)
                ft.writelines(os.path.join('images', 'train' + '2017', name_img) + '\n')

        shutil.copyfile(path_img, path_img_out)
        if os.path.exists(path_label):
            shutil.copyfile(path_label, path_label_out)


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def make_dirs(args):
    shutil.rmtree(args.dir_root_out)
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(args.dir_root_out, 'labels', subset + '2017'), exist_ok=True)
        os.makedirs(os.path.join(args.dir_root_out, 'images', subset + '2017'), exist_ok=True)


def parse_ars():
    set_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_imgs_in', default='/media/manu/kingstoo/bhv_phone/round_1/imgs', type=str)
    parser.add_argument('--dir_root_out', default='/media/manu/kingstoo/yolov5/custom_phone', type=str)
    parser.add_argument('--rate_val', default=5000 / 118287., type=float)
    return parser.parse_args()


def main():
    args = parse_ars()
    make_dirs(args)
    process(args)


if __name__ == '__main__':
    main()
