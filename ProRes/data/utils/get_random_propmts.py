import os
import random
import glob


def get_random_prompt(file_dir):
    file_list = glob.glob(os.path.join(file_dir, "*.jpg")) + glob.glob(os.path.join(file_dir, "*.png"))
    files = random.sample(file_list, 16)
    return files


if __name__ == "__main__":
    # file_dir = "datasets/low_level/enhance/lol/our485/input"
    file_dir = "datasets/low_level/derain/train/input/"
    # file_dir = "data/low_level/denoising/sidd/train_448/input/"
    # file_dir = "data/coco/train2017"
    # file_dir = "data/coco/coco_pose_256x192/coco_pose_sigma1.5and3_train2017_maxoverlap_augflip1"
    # file_dir = "data/ade20k/images/training"
    # file_dir = "/sharefs/baaivision/xinlongwang/code/uip/data/nyuv2/sync/*/"
    files = get_random_prompt(file_dir)
    for f in files:
        # print(f)
        print(os.path.basename(f).split(".")[0])
