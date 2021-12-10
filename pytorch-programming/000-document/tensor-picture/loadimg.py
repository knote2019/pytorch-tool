import torch
from PIL.Image import Image
from torchvision import transforms

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()


def image_loader(image_name, device):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def PIL_to_tensor(image, device):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
