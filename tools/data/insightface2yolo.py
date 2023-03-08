import logging
import argparse
import shutil
import os
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('manu')


def process(args):
    logger.info(args)

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

        path_subset_txt = os.path.join(args.dir_out_root, f'{subset}2017.txt')
        with open(path_subset_txt, 'w') as fss:
            for idx, path_img_r in enumerate(db.keys()):
                img_name = path_img_r.split('/')[-1]
                img_path_r_o = os.path.join('images', f'{subset}2017', img_name)
                fss.writelines(img_path_r_o + '\n')
                img_path = os.path.join(args.dir_in_root, subset, 'images', path_img_r)
                img_path_o = os.path.join(args.dir_out_root, 'images', f'{subset}2017', img_name)
                shutil.copyfile(img_path, img_path_o)
                logger.info('processing %dth/%d img %s' % (idx, len(db), img_path))
                item = db[path_img_r]
                w_img = item['w_img']
                h_img = item['h_img']
                objs = item['objs']
                lb_path_o = os.path.join(args.dir_out_root, 'labels', f'{subset}2017', img_name.replace('.jpg', '.txt'))
                with open(lb_path_o, 'w') as flb:
                    for obj in objs:
                        obj_lst = [float(v) for v in obj.split()]
                        bbox = obj_lst[:4]
                        xc = (bbox[0] + bbox[2]) / 2 / w_img
                        yc = (bbox[1] + bbox[3]) / 2 / h_img
                        w_b = (bbox[2] - bbox[0]) / w_img
                        h_b = (bbox[3] - bbox[1]) / h_img
                        line_o = f'0 {xc} {yc} {w_b} {h_b} '
                        kps = obj_lst[4:]
                        if len(kps) == 5 * 3:
                            for i in range(5):
                                kx = kps[i * 3 + 0]
                                ky = kps[i * 3 + 1]
                                kz = kps[i * 3 + 2]
                                if not (np.array([kx, ky, kz]) == -1).all():
                                    line_o += f'{kx / w_img} {ky / h_img} {kz} '
                                else:
                                    line_o += f'{kx} {ky} {kz} '
                        line_o += '\n'
                        flb.writelines(line_o)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_out_root', default='/media/manu/kingstoo/yolov5/custom_insightface', type=str)
    parser.add_argument('--dir_in_root', default='/media/manu/samsung/widerface/data/retinaface', type=str)
    parser.add_argument('--subsets', nargs='*', default=['train', 'val'], type=str)
    return parser.parse_args()


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


def main():
    args = parse_args()
    new_dirs(args)
    process(args)


if __name__ == '__main__':
    main()
