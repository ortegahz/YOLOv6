import argparse
import logging


def run(args):
    with open(args.path_in, 'r') as f:
        lines = f.readlines()
    num_of_report, num_of_skip = 0, 0
    with open(args.path_out, 'w') as f:
        for line in lines:
            line_out = line
            line_strip = line.strip()
            if line_strip[:5] == 'top: ' and num_of_report < 8:
                if num_of_skip < 8 * 2:
                    num_of_skip += 1
                    f.writelines(line_out)
                    continue
                top_name = line_strip.split('"')[1]
                line_out = line.replace(top_name, top_name + '_report')
                num_of_report += 1
            f.writelines(line_out)
    logging.info((num_of_report, num_of_skip))



def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='/home/manu/tmp/acfree.prototxt', type=str)
    parser.add_argument('--path_out', default='/home/manu/tmp/acfree_report.prototxt', type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
