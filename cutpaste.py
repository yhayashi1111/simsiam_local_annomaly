import math
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

import simsiam.loader


def cut_paste_collate_fn(batch):
    # cutPaste return 2 tuples of tuples we convert them into a list of tuples
    img_types = list(zip(*batch))
#     print(list(zip(*batch)))
    return [torch.stack(imgs) for imgs in img_types]


class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""
    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
                                                      contrast = colorJitter,
                                                      saturation = colorJitter,
                                                      hue = colorJitter)
    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img

class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)

        return super().__call__(img, augmented)

class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2,16], height=[10,25], rotation=[-45,45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation
        self.colorJitter = self.colorJitter = transforms.ColorJitter(brightness = 0.1,
                                                      contrast = 0.1,
                                                      saturation = 0.1,
                                                      hue =0.1)

    def __call__(self, img):
        #h = img.size[0]
        #w = img.size[1]

        h=224
        w=224

        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        #if self.colorJitter:
        patch = self.colorJitter(patch)


        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg,expand=True)

        #paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented)

class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)

class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)

        return org, cutpaste_normal, cutpaste_scar

class smooth_blend(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[9,38] ,**kwags):
        super(smooth_blend, self).__init__(**kwags)
        self.width = width
        self.gaussian = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(8, 8))



    def __call__(self,img):
        h = 224
        w = 224

        # cut region
        cut_w = random.randint(9,38)
        #print(cut_w)

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
        else:
            print(cut_w)


        #cut
        from_location_h = random.randint(0, h - cut_h)
        from_location_w = random.randint(0, w - cut_w)

        patch = img[:, from_location_h:from_location_h + cut_h, from_location_w:from_location_w + cut_w]
        patch = self.gaussian(patch)

        '''
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        patch_tensor = torch.tensor(np.array(patch).transpose((2, 0, 1)), dtype=torch.float32).to(self.device)
        blurred_patch_tensor = self.gaussian(patch_tensor)
        blurred_patch = Image.fromarray(blurred_patch_tensor.byte().cpu().numpy().transpose((1, 2, 0)))
        '''
        #patch = img[:,from_location_h:from_location_h+cut_h,from_location_w:from_location_w+cut_w]
        #print(patch.size())

        #patch = self.gaussian(patch)

        #from_location_h_a = random.randint(0, h - cut_h)
        #from_location_w_a = random.randint(0, w - cut_w)

        #img[:,from_location_h_a:from_location_h_a+cut_h,from_location_w_a:from_location_w_a+cut_w] = patch

        # rotate
        #rot_deg = random.uniform(*self.rotation)
        #patch = patch.convert("RGBA")#.rotate(rot_deg,expand=True)

        #paste
        to_location_h = random.randint(0, 224 - cut_h)
        to_location_w = random.randint(0, 224 - cut_w)

        '''
        mask = blurred_patch.split()[-1]
        blurred_patch = blurred_patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(blurred_patch, (to_location_w, to_location_h), mask=mask)

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        '''
        img[:, to_location_h:to_location_h + cut_h, to_location_w:to_location_w + cut_w] = patch
        #return super().__call__(img)

        return img


class Cut_blend(object):
    def __init__(self, **kwags):
        self.scar = smooth_blend(**kwags)

    def __call__(self, img):
        cutpaste_scar = self.scar(img)

        return cutpaste_scar

