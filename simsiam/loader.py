# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

from cutpaste import CutPasteScar


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, small_transform_positive, small_transform_negative, size_transform):
        self.base_transform = base_transform
        self.small_transform_positive = small_transform_positive
        self.small_transform_negative = small_transform_negative
        self.size_transform = size_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        small_positive = self.small_transform_positive(x)
        small_negative = self.small_transform_negative(x)
        base = self.size_transform(x)

        return [q, k, base,small_positive,small_negative]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Gaussian_kernel(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma_x=8,sigma_y=8):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def __call__(self, size):
        x = torch.linspace(-size // 2 + 1, size // 2, size)
        y = torch.linspace(-size // 2 + 1, size // 2, size)
        x = x.view(size, 1)
        y = y.view(1, size)

        x_component = torch.exp(- (x ** 2) / (2 * self.sigma_x ** 2))
        y_component = torch.exp(- (y ** 2) / (2 * self.sigma_y ** 2))

        kernel = x_component * y_component
        kernel = kernel / torch.sum(kernel)

        return kernel


class smooth_blend(object):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[9,38], **kwags):
        super(smooth_blend, self).__init__(**kwags)
        self.width = width
        self.gaussian = transforms.RandomApply([GaussianBlur([8., 8.])], p=1.)


    def __call__(self,img):
        h = img.size[0]
        w = img.size[1]

        # cut region
        cut_w = random.randint(9,38)

        if cut_w == 9:
            cut_h = random.randint(28,30)
        elif cut_w == 10:
            cut_h = random.randint(26,34)
        elif cut_w == 11:
            cut_h = random.randint(23,37)
        elif cut_w == 12:
            cut_h = random.randint(21,39)
        elif cut_w == 13:
            cut_h = random.randint(20,39)
        elif cut_w == 14:
            cut_h = random.randint(18,36)
        elif cut_w == 15:
            cut_h = random.randint(17,34)
        elif cut_w == 16:
            cut_h = random.randint(16,32)
        elif cut_w == 17:
            cut_h = random.randint(15,30)
        elif cut_w == 18:
            cut_h = random.randint(14,28)
        elif cut_w == 19:
            cut_h = random.randint(14,27)
        elif cut_w == 20:
            cut_h = random.randint(13,25)
        elif cut_w == 21:
            cut_h = random.randint(12,24)
        elif cut_w == 22:
            cut_h = random.randint(12,23)
        elif cut_w == 23:
            cut_h = random.randint(11,22)
        elif cut_w == 24:
            cut_h = random.randint(11,21)
        elif cut_w == 25:
            cut_h = random.randint(11,20)
        elif cut_w == 26:
            cut_h = random.randint(10,20)
        elif cut_w == 27:
            cut_h = random.randint(10,19)
        elif cut_w == 28:
            cut_h = random.randint(10,18)
        elif cut_w == 29:
            cut_h = random.randint(10,18)
        elif cut_w == 30:
            cut_h = random.randint(11,17)
        elif cut_w == 31:
            cut_h = random.randint(11,17)
        elif cut_w == 32:
            cut_h = random.randint(11,16)
        elif cut_w == 33:
            cut_h = random.randint(12,16)
        elif cut_w == 34:
            cut_h = random.randint(12,15)
        elif cut_w == 35:
            cut_h = random.randint(12,15)
        elif cut_w == 36 or cut_w == 37 or cut_w == 38:
            cut_h = 13


        #cut
        from_location_h = random.randint(0, h - cut_h)
        from_location_w = random.randint(0, w - cut_w)


        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        #patch = img[:,from_location_h:from_location_h+cut_h,from_location_w:from_location_w+cut_w]
        #print(patch.size())
        patch_np = np.array(patch)
        patch_np = cv2.GaussianBlur(patch_np, (0, 0), 8, 8)
        patch = Image.fromarray(patch_np)
        #patch = self.gaussian(patch)

        #from_location_h_a = random.randint(0, h - cut_h)
        #from_location_w_a = random.randint(0, w - cut_w)

        #img[:,from_location_h_a:from_location_h_a+cut_h,from_location_w_a:from_location_w_a+cut_w] = patch

        # rotate
        #rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA")#.rotate(rot_deg,expand=True)

        #paste
        to_location_h = random.randint(0, h - patch.size[0])
        to_location_w = random.randint(0, w - patch.size[1])

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented)

        #return img


class Cut_blend(object):
    def __init__(self, **kwags):
        self.scar = smooth_blend(**kwags)

    def __call__(self, img):
        _, cutpaste_scar = self.scar(img)

        return cutpaste_scar

