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
        type_weight_list = [1]
        # type_weight_list = [0.033, 1, 0.004, 0.029]
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


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]

        image = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])
 

        interpolation1 = 'bicubic'
        interpolation2 = 'bicubic'
            
        # no aug for instance segmentation
        if "inst" in pair['type'] and self.transforms2 is not None:
            cur_transforms = self.transforms2
        elif "pose" in pair['type'] and self.transforms3 is not None:
            cur_transforms = self.transforms3
        else:
            cur_transforms = self.transforms

        image, target = cur_transforms(image, target, interpolation1, interpolation2)
        

        valid = torch.ones_like(target)
        imagenet_mean=torch.tensor([0.485, 0.456, 0.406])
        imagenet_std=torch.tensor([0.229, 0.224, 0.225])

             
        mask = self.masked_position_generator()
        # mask all 0 
        # valid all 1 
        # Why?
        # 1 masked patch
        # 0 valid patch

        return image, target, mask, valid

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
