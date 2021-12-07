import torch
from PIL import Image
import torchvision.transforms as transforms


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(
            filter(lambda x: isinstance(x, transforms.Normalize),
                   transform.transforms))
        mean = torch.tensor(normal_transform[0].mean,
                            dtype=img_tensor.dtype,
                            device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std,
                           dtype=img_tensor.dtype,
                           device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(
        0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception(
            "Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(
                img_tensor.shape[2]))

    return img


import matplotlib.pyplot as plt

img = Image.open('./cat.jpeg')
ToTensor_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = ToTensor_transform(img)
img = transform_convert(img_tensor, ToTensor_transform)
# plt.imshow(img)
plt.savefig('./cat-new.jpeg')
