import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import random

from PIL import Image
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform
import torch.nn.functional as F

class PairDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        json_path_list: list,
        transform: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
        transform3: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        masked_position_generator: Optional[Callable] = None,
        use_two_pairs: bool = False,
        half_mask_ratio:float = 0.,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.pairs = []
        self.weights = []
        #type_weight_list = [10, 20, 2, 20, 40, 20, 2, 2, 2, 2]
        type_weight_list = [3, 3, 1, 3]
        # type_weight_list = [0.033, 1, 0.004, 0.029]
        #type_weight_list = [0.1, 0.25, 0.2, 0.2, 0.1, 0.05, 0.05, 0.05, 0.01]
        # type_weight_list = [0.1, 0.2, 0.15, 0.25, 0.2, 0.15, 0.05, 0.05, 0.01]
        #type_weight_list = [0.1, 0.15, 0.15, 0.3, 0.3, 0.2, 0.05, 0.05, 0.01]
        # type_weight_list = [0.04, 0.96]
        # derain 13712 enhance 485
        for idx, json_path in enumerate(json_path_list):
            cur_pairs = json.load(open(json_path))
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
            self.weights.extend([type_weight_list[idx] * 1./cur_num]*cur_num)
            print(json_path, type_weight_list[idx])

        #self.weights = [1./n for n in self.weights] 
        self.use_two_pairs = use_two_pairs
        if self.use_two_pairs:
            self.pair_type_dict = {}
            for idx, pair in enumerate(self.pairs):
                if "type" in pair:
                    if pair["type"] not in self.pair_type_dict:
                        self.pair_type_dict[pair["type"]] = [idx]
                    else:
                        self.pair_type_dict[pair["type"]].append(idx)
            for t in self.pair_type_dict:
                print(t, len(self.pair_type_dict[t]))

        self.transforms = PairStandardTransform(transform, target_transform) if transform is not None else None
        self.transforms2 = PairStandardTransform(transform2, target_transform) if transform2 is not None else None
        self.transforms3 = PairStandardTransform(transform3, target_transform) if transform3 is not None else None
        self.masked_position_generator = masked_position_generator
        self.half_mask_ratio = half_mask_ratio

    def _load_image(self, path: str) -> Image.Image:
        while True:
            try:
                img = Image.open(os.path.join(self.root, path))
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break
        # process for nyuv2 depth: scale to 0~255
        if "sync_depth" in path:
            # nyuv2's depth range is 0~10m
            img = np.array(img) / 10000.
            img = img * 255
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return img

    def _random_add_prompts_random_scales(self,image, prompt, prompt_range=[8,64], scale_range=[0.2,0.3]):
        image = np.asarray(image.permute(1,2,0))
        prompt = prompt

        h, w = image.shape[0],image.shape[1]
        
        mask_image = np.ones((int(h),int(w),3),dtype=np.float32)
        mask_prompt = np.zeros((int(h),int(w),3),dtype=np.float32)

        ratio = 0

        while (scale_range[0] > ratio) == True or (ratio > scale_range[1])!=True:
            h_p = w_p = int(random.uniform(prompt_range[0], prompt_range[1]))
            point_h = int(random.uniform(h_p, h-h_p))
            point_w = int(random.uniform(w_p, w-w_p))

            mask_image[point_h:point_h+h_p,point_w:point_w+w_p] = 0.0
            mask_prompt[point_h:point_h+h_p,point_w:point_w+w_p] = 1.0
            prompts_token_num = np.sum(mask_prompt)
            ratio = prompts_token_num/(h*w)

        # image = image*mask_image

        # prompt = prompt * mask_prompt
        image = image + prompt  

        return image

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]

        pair_type = pair['type']
        if 'derain' in pair_type:
            prompt = np.load('datasets/low_level/derain.npy')
        elif 'enhance' in pair_type:
            prompt = np.load('datasets/low_level/enhance.npy')
        elif 'ssid' in pair_type:
            prompt = np.load('datasets/low_level/ssid.npy')
        elif 'deblur' in pair_type:
            prompt = np.load('datasets/low_level/deblur.npy')
        
        prompt = torch.from_numpy(prompt.transpose(2, 0, 1))

        interpolation1 = 'bicubic'
        interpolation2 = 'bicubic'
        cur_transforms = self.transforms
        image = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])
        image, target = cur_transforms(image, target, interpolation1, interpolation2)

        #image = torch.from_numpy(image.transpose(2, 0, 1))


        valid = torch.ones_like(target)
        mask = self.masked_position_generator()


        return image, target, mask, valid, prompt

    def __len__(self) -> int:
        return len(self.pairs)


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target = self.transform(input, target, interpolation1, interpolation2)
        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        return input, target
