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
        type_weight_list = [1]
        # type_weight_list = [2, 3, 1, 2]
        # type_weight_list = [0.04, 0.96]
        # derain 13712 enhance 485
        for idx, json_path in enumerate(json_path_list):
            cur_pairs = json.load(open(json_path))
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
            self.weights.extend([type_weight_list[idx] * 1./cur_num]*cur_num)
            print(json_path, type_weight_list[idx])


        self.transforms = PairStandardTransform(transform, target_transform) if transform is not None else None
        self.transforms2 = PairStandardTransform(transform2, target_transform) if transform2 is not None else None
        self.transforms3 = PairStandardTransform(transform3, target_transform) if transform3 is not None else None
        self.masked_position_generator = masked_position_generator
        self.use_two_pairs = use_two_pairs
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
        img = img.convert("RGB")
        return img


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]


        image = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])


        pair_type = pair['type']
        # print(pair['image_path'],pair_type, bool('derain' in pair_type), bool('enhance' in pair_type), bool('ssid' in pair_type), bool('deblur' in pair_type))
        # exit()
        if 'derain' in pair_type:
            type_dict = torch.tensor([1, 0, 0, 0]).unsqueeze(0)
        elif 'fivek' in pair_type:
            type_dict = torch.tensor([0, 1, 0, 0]).unsqueeze(0)
        elif 'ssid' in pair_type:
            type_dict = torch.tensor([0, 0, 1, 0]).unsqueeze(0)
        elif 'deblur' in pair_type:
            type_dict = torch.tensor([0, 0, 0, 1]).unsqueeze(0)
        else:
            raise ValueError('Invalid path')

        interpolation1 = 'bicubic'
        interpolation2 = 'bicubic'
        cur_transforms = self.transforms
        image, target = cur_transforms(image, target, interpolation1, interpolation2)

        valid = torch.ones_like(target)
        mask = self.masked_position_generator()

        return image, target, mask, valid, type_dict

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
