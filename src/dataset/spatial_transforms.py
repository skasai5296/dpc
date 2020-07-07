import numbers
import random

import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


class Compose(transforms.Compose):
    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):
    def randomize_parameters(self):
        pass

class ToPILImage(transforms.ToPILImage):
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



class RandomGrayscale(transforms.RandomGrayscale):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == "L" else 3
        if self.random_p < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()

class RandomGrayscale_Nonconsistent(transforms.RandomGrayscale):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == "L" else 3
        self.randomize_parameters()
        if self.random_p < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()

class ColorJitter(transforms.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.transform = self.get_params(self.brightness, self.contrast,
                                             self.saturation, self.hue)
            self.randomize = False

        return self.transform(img)

    def randomize_parameters(self):
        self.randomize = True

class ColorJitter_Nonconsistent(object):
    """Randomly change the brightness, contrast and saturation of an image. --modified from pytorch source code
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.threshold = p

    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(name)
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        ts = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            ts.append(
                transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            ts.append(
                transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            ts.append(
                transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            ts.append(transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(ts)
        transform = transforms.Compose(ts)

        return transform

    def __call__(self, img):
        if random.random() <= self.threshold:
            transform = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
            return transform(img)
        return img

    def randomize_parameters(self):
        pass

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string


# make consistent wrt clip
class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=Image.BILINEAR,
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


# make consistent wrt clip
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
