from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import math
import random


# from https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
def shear(img, angle_to_shear, direction="x"):
    width, height = img.size
    phi = math.tan(math.radians(angle_to_shear))
    shift_in_pixels = phi * height

    if shift_in_pixels > 0:
        shift_in_pixels = math.ceil(shift_in_pixels)
    else:
        shift_in_pixels = math.floor(shift_in_pixels)

    matrix_offset = shift_in_pixels
    if angle_to_shear <= 0:
        shift_in_pixels = abs(shift_in_pixels)
        matrix_offset = 0
        phi = abs(phi) * -1

    # Note: PIL expects the inverse scale, so 1/scale_factor for example.
    transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)

    img = img.transform((int(round(width + shift_in_pixels)), height),
                            Image.AFFINE,
                            transform_matrix,
                            Image.BICUBIC)

#     img = img.crop((abs(shift_in_pixels), 0, width, height))
    return img.resize((width, height), resample=Image.BICUBIC)
    # return img


class SubPolicy0(object):
    """ Sub-policy 0 (Posterize,0.4,8) (Rotate,0.6,9) """

    def __call__(self, img, ranges):
        # print("Called Policy 0: (Posterize,0.4,8) (Rotate,0.6,9)")
        if random.random() < 0.4: img = ImageOps.posterize(img, ranges["posterize"][8])
        if random.random() < 0.6: img = img.rotate(ranges["rotate"][9])
        return img


class SubPolicy1(object):
    """ Sub-policy 1 (Solarize,0.6,5) (AutoContrast,0.6,5) """

    def __call__(self, img, ranges):
        # print("Called Policy 1: (Solarize,0.6,5) (AutoContrast,0.6,5)")
        if random.random() < 0.6: img = ImageOps.solarize(img, ranges["solarize"][5])
        if random.random() < 0.6: img = ImageOps.autocontrast(img)
        return img


class SubPolicy2(object):
    """ Sub-policy 2 (Equalize,0.8,8) (Equalize,0.6,3) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = ImageOps.equalize(img)
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class SubPolicy3(object):
    """ Sub-policy 3 (Posterize,0.6,7) (Posterize,0.6,6) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.posterize(img, ranges["posterize"][7])
        if random.random() < 0.6: img = ImageOps.posterize(img, ranges["posterize"][6])
        return img


class SubPolicy4(object):
    """ Sub-policy 4 (Equalize,0.4,7) (Solarize,0.2,4) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageOps.equalize(img)
        if random.random() < 0.2: img = ImageOps.solarize(img, ranges["solarize"][5])
        return img


class SubPolicy5(object):
    """ Sub-policy 5 (Equalize,0.4,4) (Rotate,0.8,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageOps.equalize(img)
        if random.random() < 0.8: img = img.rotate(ranges["rotate"][8])
        return img


class SubPolicy6(object):
    """ Sub-policy 6 (Solarize,0.6,3) (Equalize,0.6,7) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.solarize(img, ranges["solarize"][3])
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class SubPolicy7(object):
    """ Sub-policy 7 (Posterize,0.8,5) (Equalize,1.0,2) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = ImageOps.posterize(img, ranges["posterize"][5])
        img = ImageOps.equalize(img)
        return img


class SubPolicy8(object):
    """ Sub-policy 8 (Rotate,0.2,3) (Solarize,0.6,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.2: img = img.rotate(ranges["rotate"][3])
        if random.random() < 0.6: img = ImageOps.solarize(img, ranges["solarize"][8])
        return img


class SubPolicy9(object):
    """ Sub-policy 9 (Equalize,0.6,8) (Posterize,0.4,6) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.equalize(img)
        if random.random() < 0.4: img = ImageOps.posterize(img, ranges["posterize"][6])
        return img


class SubPolicy10(object):
    """ Sub-policy 10 (Rotate,0.8,8) (Color,0.4,0) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = img.rotate(ranges["rotate"][8])
        if random.random() < 0.4:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(ranges["color"][0])
        return img


class SubPolicy11(object):
    """ Sub-policy 11 (Rotate,0.4,9) (Equalize,0.6,2) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = img.rotate(ranges["rotate"][9])
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class SubPolicy12(object):
    """ Sub-policy 12 (Equalize,0.0,7) (Equalize,0.8,8) """

    def __call__(self, img, ranges):
        # if random.random() < 0.0: img = ImageOps.equalize(img)
        if random.random() < 0.8: img = ImageOps.equalize(img)
        return img


class SubPolicy13(object):
    """ Sub-policy 13 (Invert,0.6,4) (Equalize,1.0,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.invert(img)
        img = ImageOps.equalize(img)
        return img


class SubPolicy14(object):
    """ Sub-policy 14 (Color,0.6,4) (Contrast,1.0,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.6:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(ranges["color"][4])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(ranges["contrast"][8])
        return img


class SubPolicy15(object):
    """ Sub-policy 15 (Rotate,0.8,8) (Color,1.0,2) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = img.rotate(ranges["rotate"][8])
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(ranges["color"][2])
        return img


class SubPolicy16(object):
    """ Sub-policy 16 (Color,0.8,8) (Solarize,0.8,7) """

    def __call__(self, img, ranges):
        if random.random() < 0.8:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(ranges["color"][8])
        if random.random() < 0.8: img = ImageOps.solarize(img, ranges["solarize"][7])
        return img


class SubPolicy17(object):
    """ Sub-policy 17 (Sharpness,0.4,7) (Invert,0.6,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.4:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(ranges["sharpness"][8])
        if random.random() < 0.6: img = ImageOps.invert(img)
        return img


class SubPolicy18(object):
    """ Sub-policy 18 (ShearX,0.6,5) (Equalize,1.0,9) """

    def __call__(self, img, ranges):
        if random.random() < 0.6:
            img = shear(img, ranges["shearX"][5]*180)
        img = ImageOps.equalize(img)
        return img


class SubPolicy19(object):
    """ Sub-policy 19 (Color,0.4,0) (Equalize,0.6,3) """

    def __call__(self, img, ranges):
        if random.random() < 0.4:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(ranges["color"][0])
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class SubPolicy20(object):
    """ Sub-policy 20 (Equalize,0.4,7) (Solarize,0.2,4) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageOps.equalize(img)
        if random.random() < 0.2: img = ImageOps.solarize(img, ranges["solarize"][4])
        return img


class SubPolicy21(object):
    """ Sub-policy 21 (Solarize,0.6,5) (AutoContrast,0.6,5) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.solarize(img, ranges["solarize"][4])
        if random.random() < 0.6: img = ImageOps.autocontrast(img)
        return img


class SubPolicy22(object):
    """ Sub-policy 22 (Invert,0.6,4) (Equalize,1.0,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.invert(img)
        img = ImageOps.equalize(img)
        return img


class SubPolicy23(object):
    """ Sub-policy 23 (Color,0.6,4) (Contrast,1.0,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.6:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(ranges["color"][4])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(ranges["contrast"][8])
        return img


class AutoAugmentImageNetPolicy(object):
    def __init__(self):
        self.policies = [SubPolicy0(), SubPolicy1(), SubPolicy2(), SubPolicy3(), SubPolicy4(), SubPolicy5(),
                         SubPolicy6(), SubPolicy7(), SubPolicy8(), SubPolicy9(), SubPolicy10(), SubPolicy11(),
                         SubPolicy12(), SubPolicy13(), SubPolicy14(), SubPolicy15(), SubPolicy16(), SubPolicy17(),
                         SubPolicy18(), SubPolicy19(), SubPolicy20(), SubPolicy21(), SubPolicy22(), SubPolicy23()]
        self.ranges = {
            "shearX": np.linspace(-0.3, 0.3, 10),
            "rotate": np.linspace(-30, 30, 10),
            "color": np.linspace(0.1, 1.9, 10),
            "posterize": np.round(np.linspace(4, 8, 10), 0).astype(np.int),
            "solarize": np.linspace(0, 256, 10),
            "contrast": np.linspace(0.1, 1.9, 10),
            "sharpness": np.linspace(0.1, 1.9, 10)
        }

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img, self.ranges)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"