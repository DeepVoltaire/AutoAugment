# AutoAugment-PyTorch
Implementation of a PyTorch transform that mimics the ImageNet Augmentation Policy learned by [AutoAugment](https://arxiv.org/abs/1805.09501v1), described in this [Google AI Blogpost](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html).

![Examples of the best ImageNet Policy](Figure2_Paper.png)

# How to use
```python
from aa_imagenet import AutoAugmentImageNetPolicy
data = ImageFolder(rootdir, transform=transforms.Compose(
                        [transforms.Resize(256), transforms.RandomResizedCrop(224), 
                         transforms.RandomHorizontalFlip(),
                         AutoAugmentImageNetPolicy(), transforms.ToTensor()]))
loader = DataLoader(data, ...)
```

#### Shear function adapted from [Augmentor](# from https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py)
