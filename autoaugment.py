from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import math
import random


class ImageNetPolicy(object):
    """ Best AutoAugment Policy of 24 sub policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self):
        self.policies = [ImageNetSubPolicy0(), ImageNetSubPolicy1(), ImageNetSubPolicy2(), ImageNetSubPolicy3(), ImageNetSubPolicy4(), ImageNetSubPolicy5(),
                         ImageNetSubPolicy6(), ImageNetSubPolicy7(), ImageNetSubPolicy8(), ImageNetSubPolicy9(), ImageNetSubPolicy10(), ImageNetSubPolicy11(),
                         ImageNetSubPolicy12(), ImageNetSubPolicy13(), ImageNetSubPolicy14(), ImageNetSubPolicy15(), ImageNetSubPolicy16(), ImageNetSubPolicy17(),
                         ImageNetSubPolicy18(), ImageNetSubPolicy19(), ImageNetSubPolicy20(), ImageNetSubPolicy21(), ImageNetSubPolicy22(), ImageNetSubPolicy23()]
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


class ImageNetSubPolicy0(object):
    """ ImageNetSub-policy 0 (Posterize,0.4,8) (Rotate,0.6,9) """

    def __call__(self, img, ranges):
        # print("Called Policy 0: (Posterize,0.4,8) (Rotate,0.6,9)")
        if random.random() < 0.4: img = ImageOps.posterize(img, ranges["posterize"][8])
        if random.random() < 0.6: img = img.rotate(ranges["rotate"][9])
        return img


class ImageNetSubPolicy1(object):
    """ ImageNetSub-policy 1 (Solarize,0.6,5) (AutoContrast,0.6,5) """

    def __call__(self, img, ranges):
        # print("Called Policy 1: (Solarize,0.6,5) (AutoContrast,0.6,5)")
        if random.random() < 0.6: img = ImageOps.solarize(img, ranges["solarize"][5])
        if random.random() < 0.6: img = ImageOps.autocontrast(img)
        return img


class ImageNetSubPolicy2(object):
    """ ImageNetSub-policy 2 (Equalize,0.8,8) (Equalize,0.6,3) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = ImageOps.equalize(img)
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy3(object):
    """ ImageNetSub-policy 3 (Posterize,0.6,7) (Posterize,0.6,6) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.posterize(img, ranges["posterize"][7])
        if random.random() < 0.6: img = ImageOps.posterize(img, ranges["posterize"][6])
        return img


class ImageNetSubPolicy4(object):
    """ ImageNetSub-policy 4 (Equalize,0.4,7) (Solarize,0.2,4) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageOps.equalize(img)
        if random.random() < 0.2: img = ImageOps.solarize(img, ranges["solarize"][4])
        return img


class ImageNetSubPolicy5(object):
    """ ImageNetSub-policy 5 (Equalize,0.4,4) (Rotate,0.8,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageOps.equalize(img)
        if random.random() < 0.8: img = img.rotate(ranges["rotate"][8])
        return img


class ImageNetSubPolicy6(object):
    """ ImageNetSub-policy 6 (Solarize,0.6,3) (Equalize,0.6,7) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.solarize(img, ranges["solarize"][3])
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy7(object):
    """ ImageNetSub-policy 7 (Posterize,0.8,5) (Equalize,1.0,2) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = ImageOps.posterize(img, ranges["posterize"][5])
        img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy8(object):
    """ ImageNetSub-policy 8 (Rotate,0.2,3) (Solarize,0.6,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.2: img = img.rotate(ranges["rotate"][3])
        if random.random() < 0.6: img = ImageOps.solarize(img, ranges["solarize"][8])
        return img


class ImageNetSubPolicy9(object):
    """ ImageNetSub-policy 9 (Equalize,0.6,8) (Posterize,0.4,6) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.equalize(img)
        if random.random() < 0.4: img = ImageOps.posterize(img, ranges["posterize"][6])
        return img


class ImageNetSubPolicy10(object):
    """ ImageNetSub-policy 10 (Rotate,0.8,8) (Color,0.4,0) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = img.rotate(ranges["rotate"][8])
        if random.random() < 0.4: img = ImageEnhance.Color(img).enhance(ranges["color"][0])
        return img


class ImageNetSubPolicy11(object):
    """ ImageNetSub-policy 11 (Rotate,0.4,9) (Equalize,0.6,2) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = img.rotate(ranges["rotate"][9])
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy12(object):
    """ ImageNetSub-policy 12 (Equalize,0.0,7) (Equalize,0.8,8) """

    def __call__(self, img, ranges):
        # if random.random() < 0.0: img = ImageOps.equalize(img)
        if random.random() < 0.8: img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy13(object):
    """ ImageNetSub-policy 13 (Invert,0.6,4) (Equalize,1.0,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.invert(img)
        img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy14(object):
    """ ImageNetSub-policy 14 (Color,0.6,4) (Contrast,1.0,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.6:
            img = ImageEnhance.Color(img).enhance(ranges["color"][4])
        img = ImageEnhance.Contrast(img).enhance(ranges["contrast"][8])
        return img


class ImageNetSubPolicy15(object):
    """ ImageNetSub-policy 15 (Rotate,0.8,8) (Color,1.0,2) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = img.rotate(ranges["rotate"][8])
        img = ImageEnhance.Color(img).enhance(ranges["color"][2])
        return img


class ImageNetSubPolicy16(object):
    """ ImageNetSub-policy 16 (Color,0.8,8) (Solarize,0.8,7) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = ImageEnhance.Color(img).enhance(ranges["color"][8])
        if random.random() < 0.8: img = ImageOps.solarize(img, ranges["solarize"][7])
        return img


class ImageNetSubPolicy17(object):
    """ ImageNetSub-policy 17 (Sharpness,0.4,7) (Invert,0.6,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageEnhance.Sharpness(img).enhance(ranges["sharpness"][7])
        if random.random() < 0.6: img = ImageOps.invert(img)
        return img


class ImageNetSubPolicy18(object):
    """ ImageNetSub-policy 18 (ShearX,0.6,5) (Equalize,1.0,9) """

    def __call__(self, img, ranges):
        if random.random() < 0.6:
            img = shear(img, ranges["shearX"][5]*180, direction="x")
        img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy19(object):
    """ ImageNetSub-policy 19 (Color,0.4,0) (Equalize,0.6,3) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageEnhance.Color(img).enhance(ranges["color"][0])
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy20(object):
    """ ImageNetSub-policy 20 (Equalize,0.4,7) (Solarize,0.2,4) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageOps.equalize(img)
        if random.random() < 0.2: img = ImageOps.solarize(img, ranges["solarize"][4])
        return img


class ImageNetSubPolicy21(object):
    """ ImageNetSub-policy 21 (Solarize,0.6,5) (AutoContrast,0.6,5) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.solarize(img, ranges["solarize"][5])
        if random.random() < 0.6: img = ImageOps.autocontrast(img)
        return img


class ImageNetSubPolicy22(object):
    """ ImageNetSub-policy 22 (Invert,0.6,4) (Equalize,1.0,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.invert(img)
        img = ImageOps.equalize(img)
        return img


class ImageNetSubPolicy23(object):
    """ ImageNetSub-policy 23 (Color,0.6,4) (Contrast,1.0,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageEnhance.Color(img).enhance(ranges["color"][4])
        img = ImageEnhance.Contrast(img).enhance(ranges["contrast"][8])
        return img


class CIFAR10Policy(object):
    """ Best AutoAugment Policy of 25 sub policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self):
        self.policies = [CIFAR10SubPolicy0(), CIFAR10SubPolicy1(), CIFAR10SubPolicy2(), CIFAR10SubPolicy3(),
                         CIFAR10SubPolicy4(), CIFAR10SubPolicy5(), CIFAR10SubPolicy6(), CIFAR10SubPolicy7(),
                         CIFAR10SubPolicy8(), CIFAR10SubPolicy9(), CIFAR10SubPolicy10(), CIFAR10SubPolicy11(),
                         CIFAR10SubPolicy12(), CIFAR10SubPolicy13(), CIFAR10SubPolicy14(), CIFAR10SubPolicy15(),
                         CIFAR10SubPolicy16(), CIFAR10SubPolicy17(), CIFAR10SubPolicy18(), CIFAR10SubPolicy19(),
                         CIFAR10SubPolicy20(), CIFAR10SubPolicy21(), CIFAR10SubPolicy22(), CIFAR10SubPolicy23(),
                         CIFAR10SubPolicy24()]

        self.ranges = {
            "shearX": np.linspace(-0.3, 0.3, 10),
            "shearY": np.linspace(-0.3, 0.3, 10),
            "translateX": np.linspace(-150 / 331, 150 / 331, 10),
            "translateY": np.linspace(-150 / 331, 150 / 331, 10),
            "rotate": np.linspace(-30, 30, 10),
            "color": np.linspace(0.1, 1.9, 10),
            "posterize": np.round(np.linspace(4, 8, 10), 0).astype(np.int),
            "solarize": np.linspace(0, 256, 10),
            "contrast": np.linspace(0.1, 1.9, 10),
            "sharpness": np.linspace(0.1, 1.9, 10),
            "brightness": np.linspace(0.1, 1.9, 10)
        }

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img, self.ranges)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"



class CIFAR10SubPolicy0(object):
    """ CIFAR10 Sub-policy 0 (Invert,0.1,7) (Contrast,0.2,6) """

    def __call__(self, img, ranges):
        if random.random() < 0.1: img = ImageOps.invert(img)
        if random.random() < 0.2: img = ImageEnhance.Contrast(img).enhance(ranges["contrast"][6])
        return img


class CIFAR10SubPolicy1(object):
    """ CIFAR10 Sub-policy 1 (Rotate,0.7,2) (TranslateX,0.3,9) """

    def __call__(self, img, ranges):
        if random.random() < 0.7: img = img.rotate(ranges["rotate"][2])
        if random.random() < 0.3:
            img = img.transform(img.size, Image.AFFINE, (1, 0, ranges["translateX"][9]*img.size[0], 0, 1, 0))
        return img


class CIFAR10SubPolicy2(object):
    """ CIFAR10 Sub-policy 2 (Sharpness,0.8,1) (Sharpness,0.9,3) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = ImageEnhance.Sharpness(img).enhance(ranges["sharpness"][1])
        if random.random() < 0.9: img = ImageEnhance.Sharpness(img).enhance(ranges["sharpness"][3])
        return img


class CIFAR10SubPolicy3(object):
    """ CIFAR10 Sub-policy 3 (ShearY,0.5,8) (TranslateY,0.7,9) """

    def __call__(self, img, ranges):
        if random.random() < 0.5: img = shear(img, ranges["shearY"][8]*180, direction="y")
        if random.random() < 0.7:
            img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, ranges["translateY"][9]*img.size[1]))
        return img


class CIFAR10SubPolicy4(object):
    """ CIFAR10 Sub-policy 4 (AutoContrast,0.5,8) (Equalize,0.9,2) """

    def __call__(self, img, ranges):
        if random.random() < 0.5: img = ImageOps.autocontrast(img)
        if random.random() < 0.9: img = ImageOps.equalize(img)
        return img


class CIFAR10SubPolicy5(object):
    """ CIFAR10 Sub-policy 5 (ShearY,0.2,7) (Posterize,0.3,7) """

    def __call__(self, img, ranges):
        if random.random() < 0.2: img = shear(img, ranges["shearY"][7]*180, direction="y")
        if random.random() < 0.3: img = ImageOps.posterize(img, ranges["posterize"][7])
        return img


class CIFAR10SubPolicy6(object):
    """ CIFAR10 Sub-policy 6 (Color,0.4,3) (Brightness,0.6,7) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageEnhance.Color(img).enhance(ranges["color"][3])
        if random.random() < 0.6: img = ImageEnhance.Brightness(img).enhance(ranges["brightness"][7])
        return img


class CIFAR10SubPolicy7(object):
    """ CIFAR10 Sub-policy 7 (Sharpness,0.3,9) (Brightness,0.7,9) """

    def __call__(self, img, ranges):
        if random.random() < 0.3: img = ImageEnhance.Sharpness(img).enhance(ranges["sharpness"][9])
        if random.random() < 0.7: img = ImageEnhance.Brightness(img).enhance(ranges["brightness"][9])
        return img


class CIFAR10SubPolicy8(object):
    """ CIFAR10 Sub-policy 8 (Equalize,0.6,5) (Equalize,0.5,1) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageOps.equalize(img)
        if random.random() < 0.5: img = ImageOps.equalize(img)
        return img


class CIFAR10SubPolicy9(object):
    """ CIFAR10 Sub-policy 9 (Contrast,0.6,7) (Sharpness,0.6,5) """

    def __call__(self, img, ranges):
        if random.random() < 0.6: img = ImageEnhance.Contrast(img).enhance(ranges["contrast"][7])
        if random.random() < 0.6: img = ImageEnhance.Sharpness(img).enhance(ranges["sharpness"][5])
        return img


class CIFAR10SubPolicy10(object):
    """ CIFAR10 Sub-policy 10 (Color,0.7,7) (TranslateX,0.5,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.7: img = ImageEnhance.Color(img).enhance(ranges["color"][7])
        if random.random() < 0.5:
            img = img.transform(img.size, Image.AFFINE, (1, 0, ranges["translateX"][8] * img.size[0], 0, 1, 0))
        return img


class CIFAR10SubPolicy11(object):
    """ CIFAR10 Sub-policy 11 (Equalize,0.3,7) (AutoContrast,0.4,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.3: img = ImageOps.equalize(img)
        if random.random() < 0.4: img = ImageOps.autocontrast(img)
        return img


class CIFAR10SubPolicy12(object):
    """ CIFAR10 Sub-policy 12 (TranslateY,0.4,3) (Sharpness,0.2,6) """

    def __call__(self, img, ranges):
        if random.random() < 0.4:
            img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, ranges["translateY"][3]*img.size[1]))
        if random.random() < 0.2: img = ImageEnhance.Sharpness(img).enhance(ranges["sharpness"][5])
        return img


class CIFAR10SubPolicy13(object):
    """ CIFAR10 Sub-policy 13 (Brightness,0.9,6) (Color,0.2,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.9: img = ImageEnhance.Brightness(img).enhance(ranges["brightness"][6])
        if random.random() < 0.2: img = ImageEnhance.Color(img).enhance(ranges["color"][8])
        return img


class CIFAR10SubPolicy14(object):
    """ CIFAR10 Sub-policy 14 (Solarize,0.5,2) (Invert,0.0,3) """

    def __call__(self, img, ranges):
        if random.random() < 0.5: img = ImageOps.solarize(img, ranges["solarize"][2])
        # if random.random() < 0.0: img = ImageOps.invert(img)
        return img


class CIFAR10SubPolicy15(object):
    """ CIFAR10 Sub-policy 15 (Equalize,0.2,0) (AutoContrast,0.6,0) """

    def __call__(self, img, ranges):
        if random.random() < 0.2: img = ImageOps.equalize(img)
        if random.random() < 0.6: img = ImageOps.autocontrast(img)
        return img


class CIFAR10SubPolicy16(object):
    """ CIFAR10 Sub-policy 16 (Equalize,0.2,8) (Equalize,0.6,4) """

    def __call__(self, img, ranges):
        if random.random() < 0.2: img = ImageOps.equalize(img)
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class CIFAR10SubPolicy17(object):
    """ CIFAR10 Sub-policy 17 (Color,0.9,9) (Equalize,0.6,6) """

    def __call__(self, img, ranges):
        if random.random() < 0.9: img = ImageEnhance.Color(img).enhance(ranges["color"][9])
        if random.random() < 0.6: img = ImageOps.equalize(img)
        return img


class CIFAR10SubPolicy18(object):
    """ Sub-policy 18 (AutoContrast,0.8,4) (Solarize,0.2,8) """

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = ImageOps.autocontrast(img)
        if random.random() < 0.2: img = ImageOps.solarize(img, ranges["solarize"][8])
        return img


class CIFAR10SubPolicy19(object):
    """ CIFAR10 Sub-policy 19 (Brightness,0.1,3) (Color,0.7,0) """

    def __call__(self, img, ranges):
        if random.random() < 0.1: img = ImageEnhance.Brightness(img).enhance(ranges["brightness"][3])
        if random.random() < 0.7: img = ImageEnhance.Color(img).enhance(ranges["color"][0])
        return img


class CIFAR10SubPolicy20(object):
    """ CIFAR10 Sub-policy 20 (Solarize,0.4,5) (AutoContrast,0.9,3) """

    def __call__(self, img, ranges):
        if random.random() < 0.4: img = ImageOps.solarize(img, ranges["solarize"][5])
        if random.random() < 0.9: img = ImageOps.autocontrast(img)
        return img


class CIFAR10SubPolicy21(object):
    """ CIFAR10 Sub-policy 21 (TranslateY,0.9,9) (TranslateY,0.7,9) """

    def __call__(self, img, ranges):
        if random.random() < 0.9:
            img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, ranges["translateY"][9] * img.size[1]))
        if random.random() < 0.7:
            img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, ranges["translateY"][9] * img.size[1]))
        return img


class CIFAR10SubPolicy22(object):
    """ CIFAR10 Sub-policy 22 (AutoContrast,0.9,2) (Solarize,0.8,3) """

    def __call__(self, img, ranges):
        if random.random() < 0.9: img = ImageOps.autocontrast(img)
        if random.random() < 0.8: img = ImageOps.solarize(img, ranges["solarize"][3])
        return img


class CIFAR10SubPolicy23(object):
    """ CIFAR10 Sub-policy 23 (Equalize,0.8,8) (Invert,0.1,3)"""

    def __call__(self, img, ranges):
        if random.random() < 0.8: img = ImageOps.equalize(img)
        if random.random() < 0.1: img = ImageOps.invert(img)
        return img


class CIFAR10SubPolicy24(object):
    """ CIFAR10 Sub-policy 24 (TranslateY,0.7,9) (AutoContrast,0.9,1)"""

    def __call__(self, img, ranges):
        if random.random() < 0.7:
            img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, ranges["translateY"][9] * img.size[1]))
        if random.random() < 0.9: img = ImageOps.autocontrast(img)
        return img


# from https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
def shear(img, angle_to_shear, direction="x"):
    width, height = img.size
    phi = math.tan(math.radians(angle_to_shear))

    if direction=="x":
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

    elif direction == "y":
        shift_in_pixels = phi * width

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, 0, 0, phi, 1, -matrix_offset)

        image = img.transform((width, int(round(height + shift_in_pixels))),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

        # image = image.crop((0, abs(shift_in_pixels), width, height))

        return image.resize((width, height), resample=Image.BICUBIC)