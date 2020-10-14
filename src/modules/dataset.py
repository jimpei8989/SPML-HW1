from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tf
from PIL import Image

from modules.utils import all_labels

cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465])
cifar10_std = torch.tensor([0.2023, 0.1994, 0.2010])


class ImageDataset(Dataset):
    '''
    Each item in `data` is a tuple (image_name, image, label), where
        - image_name is a string, e.g. 'airplane1.png'
        - image is a torch.Tensor / PIL instance
        - label is a integer in [0, 9], representing the label
    '''
    def __init__(self, data_dir: Optional[Path] = None, transform = None):
        self._data_dir = Path(data_dir)
        self._data = []
        self._transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        image_name, image, label = self._data[index]
        if isinstance(image, Image.Image):
            image = tf.to_tensor(image)
        if self._transform:
            image = self._transform(image)
        return image_name, image.cuda(), torch.tensor(label).cuda()

    def get_np_images(self):
        for _, image, _ in self._data:
            # change type to `np.int32` to prevent overflow
            yield np.asarray(image).astype(np.int32)


class OriginalDataset(ImageDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_from_directory()

    def load_from_directory(self):
        assert self._data_dir.is_dir(), "OriginalDataset.data_dir should be a directory"

        for label, label_name in enumerate(all_labels):
            label_dir = self._data_dir / label_name

            assert label_dir.is_dir(), "OriginalDataset.data_dir should contain directory `label_name`"

            for i in range(1, 11):
                image_name = f'{label_name}{i}.png'
                self._data.append((image_name, Image.open(label_dir / image_name), label))


class AdversarialDataset(ImageDataset):
    def __init__(self, data_dir: Optional[Path] = None, transform = None):
        super().__init__(data_dir=data_dir, transform=transform)

    def add(self, image_name, image, label):
        # The added things could be unsqueezed (with the first batch dimension)
        if isinstance(image_name, tuple) and len(image_name) == 1:
            image_name = image_name[0]

        if len(image.shape) == 4:
            image = torch.squeeze(image, dim=0)

        if isinstance(label, torch.LongTensor):
            label = int(label)

        # Move to CPU
        if image.is_cuda:
            image = image.cpu()

        self._data.append((image_name, image, label))

    def save_to_directory(self):
        if not self._data_dir.is_dir():
            self._data_dir.mkdir()

        for label, label_name in enumerate(all_labels):
            label_dir = self._data_dir / label_name
            if not label_dir.is_dir():
                label_dir.mkdir()

            for image_name, image, label in filter(lambda d: d[2] == label, self._data):
                if isinstance(image, torch.Tensor):
                    image = tf.to_pil_image(image)
                image.save(label_dir / image_name)
