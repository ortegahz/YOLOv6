import argparse
import glob
import logging
import os
import shutil

from tqdm import tqdm


def process(args):
    for dir_idx, dir_root in enumerate(args.dirs_root_in):
        paths_img = glob.glob(os.path.join(dir_root, '*.jpg'))  # single img format

        for path_img in tqdm(paths_img):
            name_img = os.path.basename(path_img)
            name_img_out = name_img[:-4] + f'_{dir_idx}' + name_img[-4:]
            path_img_out = os.path.join(args.dir_root_out, name_img_out)
            assert not os.path.exists(path_img_out)
            shutil.copyfile(path_img, path_img_out)


def make_dirs(args):
    if os.path.exists(args.dir_root_out):
        shutil.rmtree(args.dir_root_out)
    os.makedirs(os.path.join(args.dir_root_out), exist_ok=True)


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs_root_in', nargs='+',
                        default=['/media/manu/kingstoo/tmp/jjl/1',
                                 '/media/manu/kingstoo/tmp/jjl/2',
                                 '/media/manu/kingstoo/tmp/jjl/3',
                                 '/media/manu/kingstoo/tmp/jjl/4',
                                 '/media/manu/kingstoo/tmp/jjl/5',
                                 '/media/manu/kingstoo/tmp/jjl/6'],
                        type=str)
    parser.add_argument('--dir_root_out', default='/media/manu/kingstoo/tmp/jjl/m', type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    make_dirs(args)
    process(args)


if __name__ == '__main__':
    main()
