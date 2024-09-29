import os
import glob
import json
import tqdm


if __name__ == "__main__":
    # type = 'gt_sub_input'
    type = 'groundtruth'

    # split = 'train'
    split = 'train'
    if split == 'train':
        image_dir = "./datasets/low_level/deblur/train/input"
        save_path = "./datasets/low_level/{}-deblur_gopro_train.json".format(type)
    elif split == 'val':
        image_dir = "./datasets/low_level/deblur/test/RealBlur_J/input"
        save_path = "./datasets/low_level/{}-deblur_realblur_j_val.json".format(type)
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
        pair_dict["type"] = "{}_deblur".format(type)
        output_dict.append(pair_dict)

    json.dump(output_dict, open(save_path, 'w'))
