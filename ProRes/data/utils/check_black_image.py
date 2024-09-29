import copy
import os
import glob
import json
import warnings
import argparse
import shutil

import numpy as np
import tqdm
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--tgt_dir', type=str, help='dir to ckpt', required=True)
    parser.add_argument('--remove', action="store_true", help='dir to ckpt', default=False)
    return parser.parse_args()


def load_image_with_retry(image_path):
    while True:
        try:
            img = Image.open(image_path)
            return img
        except OSError as e:
            print(f"Catched exception: {str(e)}. Re-trying...")
            import time
            time.sleep(1)


if __name__ == '__main__':
    args = get_args_parser()
    tgt_dir = args.tgt_dir

    image_list = glob.glob(os.path.join(tgt_dir, "*.png")) + glob.glob(os.path.join(tgt_dir, "*.jpg"))
    num_black = 0
    for image_path in tqdm.tqdm(image_list):
        image = load_image_with_retry(image_path)
        image = np.array(image)
        if (image == 0).all():
            num_black += 1
            print("{}.  {} is black!".format(num_black, image_path))
            if args.remove:
                os.remove(image_path)

    print("num black: {}".format(num_black))
