import os
import glob
import tqdm
from PIL import Image
import torchvision.transforms.functional as transform_func


if __name__ == '__main__':
    image_src_dir = "coco/train2017"
    image_tgt_dir = "coco/train2017_flip"

    if not os.path.exists(image_tgt_dir):
        os.makedirs(image_tgt_dir)

    image_list = glob.glob(os.path.join(image_src_dir, "*.jpg"))
    for image_path in tqdm.tqdm(image_list):
        image = Image.open(image_path)
        image_flip = transform_func.hflip(image)

        file_name = os.path.basename(image_path).replace(".jpg", "_flip.jpg")
        tgt_path = os.path.join(image_tgt_dir, file_name)
        image_flip.save(tgt_path)