import numpy as np
import torch
from skimage.transform import rotate
import random
import cv2
import numbers
import collections
from torchvision import transforms
from PIL import Image

def random_five_crop(img, size):
    random_crop = transforms.RandomCrop(size)
    a = random_crop(img)
    b = random_crop(img)
    c = random_crop(img)
    d = random_crop(img)
    e = random_crop(img)
    return (a, b, c, d, e)
class RandomTenCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        first_five = random_five_crop(img, self.size)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        second_five = random_five_crop(img, self.size)
        return first_five + second_five
    def __repr__(self):
        return self.__class__.__name__ + '()'
class RandomRotate(object):
    def __init__(self, angs):
        self.angs = angs
    def __call__(self, pic):
        ang = random.choice(self.angs)
        return rotate(pic, ang)
    def __repr__(self):
        return self.__class__.__name__ + '()'
class CenterCrop(object):
    def __init__(self, size):
        self.crop_size = size
    def __call__(self, x):#x's shape is [n, m, 3]
        assert x.ndim == 3
        centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
        halfw, halfh = self.crop_size[0] // 2, self.crop_size[1] // 2
        assert halfh <= centerh and halfw <= centerw
        return x[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]
    def __repr__(self):
        return self.__class__.__name__ + '()'
class RandomCrop(object):
    def __init__(self, size):
        self.crop_size = size
    def __call__(self, x):## x's shape is [n, m, 3]
        assert x.ndim == 3
        assert x.shape[0] > self.crop_size[0] and x.shape[1] > self.crop_size[1]
        startw = np.random.randint(0, x.shape[0]-self.crop_size[0])
        starth = np.random.randint(0, x.shape[1]-self.crop_size[1])
        return x[startw: startw+self.crop_size[0], starth: starth+self.crop_size[1]]
    def __repr__(self):
        return self.__class__.__name__ + '()'
class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return cv2.resize(img, dsize=self.size)
    def __repr__(self):
        return self.__class__.__name__ + '()'
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        image -= self.mean
        image /= self.std
        return image
class ToTensor(object):
    def __call__(self, x):
        assert x.ndim == 3
        x = x.transpose((2, 0, 1))
        return torch.from_numpy(x).float() / 255.

class GammaCorrection(object):
    def __init__(self, gammas):
        self.gammas = gammas
    def __call__(self, img):
        gamma = random.choice(self.gammas)
        img = np.uint8(cv2.pow(img / 255., gamma) * 255.)
        return img

class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        return img
class HorizontalFlip(object):
    def __call__(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
        return img

def saturation(pic, factor):
    grayimg = grayscale(pic)
    img = pic * factor + grayimg * (1.0 - factor)
    img = img.astype(pic.dtype)
    return img

def brightness(pic, factor):
    img = pic * factor
    return img
def grayscale(pic):
    assert (len(pic.shape) == 3) and (pic.shape[2] == 3), "input img for grayscale() should be H*W*3 ndarray"
    grayimg = 0.299 * pic[:, :, 2] + 0.587 * pic[:, :, 1] + 0.114 * pic[:, :, 0]
    grayimg = np.repeat(grayimg[:, :, np.newaxis], 3, axis=2)
    return grayimg

def contrast(pic, factor):
    grayimg = grayscale(pic)
    ave = grayimg[:, :, 0].mean()
    ave_img = np.ndarray(shape=pic.shape, dtype=float)
    ave_img.fill(ave)
    img = pic * factor + ave_img * (1 - factor)
    return img
class ColorJitter(object):
    """
    do ColorJitter for BGR ndarray image
    factor should be a number of list of three number, all numbers should be in (0,1)
    """

    def __init__(self, factor):
        if isinstance(factor, numbers.Number) and 0 < factor < 1:
            self.saturation_factor = factor
            self.brightness_factor = factor
            self.contrast_factor = factor
        elif isinstance(factor, collections.Iterable) and len(factor) == 3 \
                and isinstance(factor[0], numbers.Number) and 0 < factor[0] < 1 \
                and isinstance(factor[1], numbers.Number) and 0 < factor[1] < 1 \
                and isinstance(factor[2], numbers.Number) and 0 < factor[2] < 1:
            self.saturation_factor = factor[0]
            self.brightness_factor = factor[1]
            self.contrast_factor = factor[2]
        else:
            raise (RuntimeError("ColorJitter factor error.\n"))

    def __call__(self, img):
        ori_type = img.dtype
        img.astype('float32')
        this_saturation_factor = 1.0 + self.saturation_factor * random.uniform(-1.0, 1.0)
        this_brightness_factor = 1.0 + self.brightness_factor * random.uniform(-1.0, 1.0)
        this_contrast_factor = 1.0 + self.contrast_factor * random.uniform(-1.0, 1.0)
        funclist = [(saturation, this_saturation_factor),
                    (brightness, this_brightness_factor),
                    (contrast, this_contrast_factor)]
        random.shuffle(funclist)
        for func in funclist:
            img = (func[0])(img, func[1])
        if ori_type == np.uint8:
            img = np.clip(img, 0, 255)
            img.astype('uint8')
        elif ori_type == np.uint16:
            img = np.clip(img, 0, 65535)
            img.astype('uint16')
        return img