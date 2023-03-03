import logging
import argparse
import shutil
import os
import cv2

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('manu')


def process(args):
    logger.info(args)

    name_windows = 'results'
    cv2.namedWindow(name_windows, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_windows, 960, 540)

    for subset in args.subsets:
        path_in_label = os.path.join(args.dir_in_root, subset, 'labelv2.txt')
        with open(path_in_label, 'r') as fl:
            lines = fl.readlines()
        db = dict()
        path_img_r = None
        for line in lines:
            if line.startswith('#'):
                line_lst = line[1:].split()
                path_img_r = line_lst[0]
                w_img = int(line_lst[1])
                h_img = int(line_lst[2])
                db[path_img_r] = dict(w_img=w_img, h_img=h_img, objs=[])
                continue
            assert path_img_r is not None
            assert path_img_r in db
            db[path_img_r]['objs'].append(line)
        logger.info('number of imgs --> %d' % len(db))

        for idx, path_img_r in enumerate(db.keys()):
            img_path = os.path.join(args.dir_in_root, subset, 'images', path_img_r)
            img = cv2.imread(img_path)
            logger.info('processing %dth/%d img %s' % (idx, len(db), img_path))
            item = db[path_img_r]
            # w_img = item['w_img']
            # h_img = item['h_img']
            objs = item['objs']
            for obj in objs:
                obj_lst = [float(v) for v in obj.split()]
                bbox = obj_lst[:4]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                kps = obj_lst[4:]
                if not len(kps) == 3 * 5:
                    continue
                kps_x, kps_y = kps[::3], kps[1::3]
                for kp_x, kp_y in zip(kps_x, kps_y):
                    if kp_x < 0 or kp_y < 0:
                        continue
                    cv2.circle(img, (int(kp_x), int(kp_y)), 3, (255, 0, 0), 2)
            cv2.imshow(name_windows, img)
            cv2.waitKey(1000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in_root', default='/media/manu/samsung/widerface/data/retinaface', type=str)
    parser.add_argument('--subsets', nargs='*', default=['train'], type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    process(args)


if __name__ == '__main__':
    main()
