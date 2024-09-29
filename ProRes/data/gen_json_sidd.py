import os
import glob
import json
import tqdm


if __name__ == "__main__":
    # type = 'gt_sub_input'
    type = 'groundtruth'

    # split = 'train'
    split = 'val'
    if split == 'train':
        image_dir = "./datasets/low_level/denoising/sidd/train_448/input"
        save_path = "./datasets/low_level/{}-denoise_ssid_train448.json".format(type)
    elif split == 'val':
        image_dir = "./datasets/low_level/denoising/sidd/sidd_val_patch256/input"
        save_path = "./datasets/low_level/{}-denoise_ssid_val256.json".format(type)
    else:
        raise NotImplementedError
    print(save_path)

    output_dict = []

    image_path_list = glob.glob(os.path.join(image_dir, '*.png'))
    for image_path in tqdm.tqdm(image_path_list):
        # image_name = os.path.basename(image_path)
        target_path = image_path.replace('input', type)
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        pair_dict = {}
        pair_dict["image_path"] = image_path
        pair_dict["target_path"] = target_path
        pair_dict["type"] = "{}_denoise_ssid_448".format(type)
        output_dict.append(pair_dict)

    json.dump(output_dict, open(save_path, 'w'))
