import random

from PIL import Image

import torchvision.transforms.functional as F
from torchvision import transforms


class Compose(transforms.Compose):
    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):
    def randomize_parameters(self):
        pass


class Normalize(transforms.Normalize):
    def randomize_parameters(self):
        pass


class Resize(transforms.Resize):
    def randomize_parameters(self):
        pass


class CenterCrop(transforms.CenterCrop):
    def randomize_parameters(self):
        pass


class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.transform = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
            self.randomize = False

        return self.transform(img)

    def randomize_parameters(self):
        self.randomize = True


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=Image.BILINEAR,
    ):

        super().__init__(size, scale, ratio, interpolation)
        self.randomize_parameters()

    def __call__(self, img):

        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False

        i, j, h, w = self.random_crop
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.randomize = True


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.random_p < self.p:
            return F.hflip(img)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()
