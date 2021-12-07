import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 2, 3], [3, 4, 5]])

print(a)
print(b)
print(torch.mm(a, b))
