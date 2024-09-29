import os
import glob
import json
import tqdm


if __name__ == "__main__":

    images_list = os.listdir("/horizon-bucket/BasicAlgorithm/Users/jiaqi.ma/fivek/input/")
    print(len(images_list))
 
    images_list.sort()

    train_images_list = images_list[:4500]
    test_images_list = images_list[4500:]

    type = 'expertC_gt'


    output_dict_train = []

    for image_path in tqdm.tqdm(train_images_list):
        # image_name = os.path.basename(image_path)
        image_dir = "./datasets/low_level/enhance/fivek/input/"
        image_path = os.path.join(image_dir, image_path)
        target_path = image_path.replace('input', 'expertC_gt')
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        pair_dict = {}
        pair_dict["image_path"] = image_path
        pair_dict["target_path"] = target_path
        pair_dict["type"] = "{}_fivek".format(type)
        output_dict_train.append(pair_dict)
    save_path_train = "./datasets/low_level/{}-fivek_train.json".format(type)
    json.dump(output_dict_train, open(save_path_train, 'w'))


    output_dict_test = []
    for image_path in tqdm.tqdm(test_images_list):
        # image_name = os.path.basename(image_path)
        image_dir = "./datasets/low_level/enhance/fivek/input/"
        image_path = os.path.join(image_dir, image_path)
        target_path = image_path.replace('input', 'expertC_gt')
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        pair_dict = {}
        pair_dict["image_path"] = image_path
        pair_dict["target_path"] = target_path
        pair_dict["type"] = "{}_fivek".format(type)
        output_dict_test.append(pair_dict)
    save_path_test = "./datasets/low_level/{}-fivek_test.json".format(type)
    json.dump(output_dict_test, open(save_path_test, 'w'))