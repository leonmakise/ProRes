"""
get subset for quick evaluation
"""
import os
import glob
import json
import tqdm
import shutil


if __name__ == "__main__":
    file_path = "coco/panoptic_val2017.json"

    data = json.load(open(file_path, 'r'))
    annotations = data['annotations']  # panoptic annos are saved in per image style
    categories = {category['id']: category for category in data['categories']}

    # note this includes crowd
    num_inst_list = []
    for anno in annotations:
        num_inst = 0
        segments_info = anno['segments_info']
        for seg in segments_info:
            if seg['iscrowd']:
                continue
            if not categories[seg['category_id']]['isthing']:
                continue
            num_inst += 1
        # if num_inst != 90:
        num_inst_list.append(num_inst)

    print(max(num_inst_list))



