import os
import glob
import json
import tqdm


if __name__ == "__main__":
    type = 'GT'
    # type = 'gt_sub_input'

    # split = 'train'
    split = 'test'
    if split == 'train':
        image_dir = "./datasets/low_level/dehaze/train/hazy/"
        save_path = "./datasets/low_level/{}-dehaze_train.json".format(type)
    elif split == 'test':
        image_dir = "./datasets/low_level/dehaze/test/hazy/"
        save_path = "./datasets/low_level/{}-dehaze_test.json".format(type)
    else:
        raise NotImplementedError
    print(save_path)

    output_dict = []

    image_path_list = glob.glob(os.path.join(image_dir, '*.jpg'))+glob.glob(os.path.join(image_dir, '*.png'))
    for image_path in tqdm.tqdm(image_path_list):
        # image_name = os.path.basename(image_path)
        target_path = image_path.replace('hazy', 'GT')
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        pair_dict = {}
        pair_dict["image_path"] = image_path
        pair_dict["target_path"] = target_path
        pair_dict["type"] = "{}_dehaze".format(type)
        output_dict.append(pair_dict)

    json.dump(output_dict, open(save_path, 'w'))
