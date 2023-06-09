import argparse
import logging
import glob
import os
import cv2

from tqdm import tqdm


def process(args):
    logging.info(args)

    n_points = 0
    for subset in args.subsets:

        paths_img = glob.glob(os.path.join(args.dir_root, 'images', subset, '*'))

        for path_img in tqdm(paths_img):
            path_label = path_img.replace('.jpg', '.txt')
            path_label = path_label.replace('images', 'labels')

            img = cv2.imread(path_img)
            h_img, w_img, _ = img.shape

            if os.path.exists(path_label):
                with open(path_label, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    l, xc, yc, w, h = line.split()
                    l, xc, yc, w, h = int(l), float(xc), float(yc), float(w), float(h)
                    color = (0, 255, 0) if l == 0 else (0, 0, 255)
                    p1 = (int((xc - w / 2) * w_img), int((yc - h / 2) * h_img))
                    p2 = (int((xc + w / 2) * w_img), int((yc + h / 2) * h_img))
                    cv2.rectangle(img, p1, p2, color, 2)
                    n_points += 1

            if not args.show:
                continue

            cv2.imshow('results', img)

            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    logging.info(f'n_points --> {n_points}')


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_ars():
    set_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root', default='/media/manu/kingstoo/tmp/pics_phone_sample', type=str)
    parser.add_argument('--subsets', nargs='+', default=['train', 'val'], type=str)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def main():
    args = parse_ars()
    process(args)


if __name__ == '__main__':
    main()
