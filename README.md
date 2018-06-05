# AutoAugment-PyTorch
Implementation of a PyTorch transform that mimics the ImageNet Augmentation Policy learned by [AutoAugment](https://arxiv.org/abs/1805.09501v1), described in this [Google AI Blogpost](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html).

![Examples of the best ImageNet Policy](Figure2_Paper.png)

# Example of how to use
```python
from aa_imagenet import AutoAugmentImageNetPolicy
data = ImageFolder(rootdir, transform=transforms.Compose(
                        [transforms.Resize(256), transforms.RandomResizedCrop(224), 
                         transforms.RandomHorizontalFlip(), AutoAugmentImageNetPolicy(), 
			 transforms.ToTensor(), transforms.Normalize(...)]))
loader = DataLoader(data, ...)
```

From the paper it is not exactly clear in what exact order to apply the preprocessing:

> For baseline augmentation, we use the standard Inception-style pre-processing which involves scaling pixel values to [-1,1],
> horizontal flips with 50% probability, and random distortions of colors. For models trained with AutoAugment, we use the baseline pre-processing
> and the policy learned on ImageNet. We find that removing the random distortions of color does not change the results for AutoAugment.

#### Shear function adapted from [Augmentor](https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py)
