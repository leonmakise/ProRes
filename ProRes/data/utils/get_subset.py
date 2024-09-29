"""
get subset for quick evaluation
"""
import os
import glob
import json
import tqdm
import shutil


def get_json_subset(file_path, output_path, num_keep=500):
    data = json.load(open(file_path, 'r'))
    # keys in data: dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    data['images'] = data['images'][:num_keep]
    with open(output_path, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    num_keep = 500
    file_path = "coco/annotations/instances_val2017.json"
    output_path = file_path.replace(".json", "_first{}.json".format(num_keep))
    if not os.path.exists(output_path):
        get_json_subset(file_path, output_path, num_keep)

    data_of_interest = json.load(open(output_path, 'r'))
    images = data_of_interest['images']
    images_list = [img['file_name'] for img in images]

    # images_src_dir = "/sharefs/wwen/unified-vp/uip/models_inference/uip_rpe_vit_large_patch16_input640_win_dec64_8glb_lr1e-3_clip1.5_bs2x8x16_maeinit_mask392_depth_ade20k_cocomask_cocoins_cocosem_cocopose_bidi_new_nearest_25ep/" \
    #                  "pano_inst_inference_epoch3_000000466730"
    images_src_dir = '/sharefs/wwen/unified-vp/uip/models_inference/' \
                     'uip_rpe_vit_large_patch16_input640_win_dec64_8glb_lr1e-3_clip1.5_bs2x8x16_maeinit_mask392_depth_ade20k_cocomask_cocoins_cocosem_cocopose_wobidi_new_nearest_50ep_insworg_newweight_posex50_insx10/' \
                     'pano_semseg_inference_epoch42_000000443397'
    # 'pano_inst_inference_epoch42_000000443397'
    assert images_src_dir[-1] != '/'
    images_tgt_dir = images_src_dir + "_first{}".format(num_keep)
    if not os.path.exists(images_tgt_dir):
        os.makedirs(images_tgt_dir)
    print(images_tgt_dir)

    image_path_list = glob.glob(os.path.join(images_src_dir, '*.png'))
    for image_path in tqdm.tqdm(image_path_list):
        # file_name = os.path.basename(image_path).split("_")[0] + ".jpg"
        file_name = os.path.basename(image_path).replace(".png", ".jpg")
        if file_name in images_list:
            shutil.copy(image_path, images_tgt_dir)

