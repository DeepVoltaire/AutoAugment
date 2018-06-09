from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import math
import random


class ImageNetPolicy(object):
    """ Best AutoAugment Policy of 24 Sub-policies on ImageNet.

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
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8),
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Best AutoAugment Policy of 25 Sub-policies on CIFAR10.

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
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Best AutoAugment Policy of 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2):
        ranges = {
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
            "brightness": np.linspace(0.1, 1.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": lambda img, magnitude: shear(img, magnitude * 180, direction="x"),
            "shearY": lambda img, magnitude: shear(img, magnitude * 180, direction="y"),
            "translateX": lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                               (1, 0, magnitude*img.size[0], 0, 1, 0)),
            "translateY": lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                               (1, 0, 0, 0, 1, magnitude*img.size[1])),
            "rotate": lambda img, magnitude: img.rotate(magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(magnitude),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(magnitude),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(magnitude),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(magnitude),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
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