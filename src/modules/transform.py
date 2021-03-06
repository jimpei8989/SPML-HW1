from functools import partial
from io import BytesIO
from typing import List, Optional

import torch

from PIL import Image
from PIL import ImageFilter as IF
from torchvision import transforms as T


def build_transform(transform):
    return {
        'Gaussian': GaussianBlur,
        'JPEG-80': partial(JpegCompression, quality=80),
        'JPEG-60': partial(JpegCompression, quality=60),
    }[transform]()


def build_transforms(transforms: Optional[List[str]]):
    if not transforms:
        return None
    return T.Compose([build_transform(t) for t in transforms])


class GaussianBlur():
    def __init__(self, r=1):
        self.filter = IF.GaussianBlur(r)

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(self.filter)


class JpegCompression():
    def __init__(self, quality=60):
        self._quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        with BytesIO() as f:
            img.save(f, format='JPEG', quality=self._quality)
            img = Image.open(f).convert('RGB')
        return img
