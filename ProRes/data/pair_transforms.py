import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torchvision.transforms as transforms

try:
    import accimage
except ImportError:
    accimage = None

import torchvision.transforms.functional as F
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode

__all__ = [
    "Compose",
    "ToTensor",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
]



class Compose(transforms.Compose):
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, tgt, interpolation1=None, interpolation2=None):
        for t in self.transforms:
            img, tgt = t(img, tgt)
        return img, tgt


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.
    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic1, pic2, interpolation1=None, interpolation2=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic1), F.to_tensor(pic2)


class Normalize(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, tensor1: Tensor, tensor2: Tensor, interpolation1=None, interpolation2=None):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor1, self.mean, self.std, self.inplace), F.normalize(tensor2, self.mean, self.std, self.inplace)


class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop a random portion of image and resize it to a given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.
    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
    ):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if interpolation1 == 'nearest':
            interpolation1 = InterpolationMode.NEAREST
        else:
            interpolation1 = InterpolationMode.BICUBIC
        if interpolation2 == 'nearest':
            interpolation2 = InterpolationMode.NEAREST
        else:
            interpolation2 = InterpolationMode.BICUBIC
            
        return F.resized_crop(img, i, j, h, w, self.size, interpolation1), \
                F.resized_crop(tgt, i, j, h, w, self.size, interpolation2)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(tgt)
        return img, tgt

