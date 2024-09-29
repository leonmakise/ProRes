import os
import glob
import json
import tqdm


if __name__ == "__main__":

    type = 'target'

    split = 'train'
    # split = 'test'
    if split == 'train':
        image_dir = "./datasets/low_level/defocusblur/dpdd/train/inputC/"
        save_path = "./datasets/low_level/{}-defocusblur_dpdd_train.json".format(type)
    elif split == 'test':
        image_dir = "./datasets/low_level/defocusblur/dpdd/test/inputC/"
        save_path = "./datasets/low_level/{}-defocusblur_dpdd_test.json".format(type)
    else:
        raise NotImplementedError
    print(save_path)

    output_dict = []

    image_path_list = glob.glob(os.path.join(image_dir, '*.png'))
    for image_path in tqdm.tqdm(image_path_list):
        # image_name = os.path.basename(image_path)
        target_path = image_path.replace('inputC', type)
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        pair_dict = {}
        pair_dict["image_path"] = image_path
        pair_dict["target_path"] = target_path
        pair_dict["type"] = "{}_defocusblur_dpdd".format(type)
        output_dict.append(pair_dict)


    image_path_list = glob.glob(os.path.join(image_dir.replace('inputC', 'inputL'), '*.png'))
    for image_path in tqdm.tqdm(image_path_list):
        # image_name = os.path.basename(image_path)
        target_path = image_path.replace('inputL', type)
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        pair_dict = {}
        pair_dict["image_path"] = image_path
        pair_dict["target_path"] = target_path
        pair_dict["type"] = "{}_defocusblur_dpdd".format(type)
        output_dict.append(pair_dict)


    image_path_list = glob.glob(os.path.join(image_dir.replace('inputC', 'inputR'), '*.png'))
    for image_path in tqdm.tqdm(image_path_list):
        # image_name = os.path.basename(image_path)
        target_path = image_path.replace('inputR', type)
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        pair_dict = {}
        pair_dict["image_path"] = image_path
        pair_dict["target_path"] = target_path
        pair_dict["type"] = "{}_defocusblur_dpdd".format(type)
        output_dict.append(pair_dict)
    json.dump(output_dict, open(save_path, 'w'))