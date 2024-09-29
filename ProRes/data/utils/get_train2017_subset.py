"""
get subset for quick evaluation
"""
import os
import glob
import json
import tqdm
import shutil


if __name__ == "__main__":
    images_src_dir = "coco/train2017_copy"
    image_path_list = glob.glob(os.path.join(images_src_dir, '*.jpg'))
    num_images = len(image_path_list)

    images_tgt_dir = images_src_dir + "_{}".format(num_images // 2)
    if not os.path.exists(images_tgt_dir):
        os.makedirs(images_tgt_dir)
    else:
        raise NotImplementedError("{} exist!".format(images_tgt_dir))
    print(images_tgt_dir)

    for image_path in tqdm.tqdm(image_path_list):
        images_tgt_path = os.path.join(images_tgt_dir, os.path.basename(image_path))
        assert not os.path.isfile(images_tgt_path)
        shutil.move(image_path, images_tgt_dir)

