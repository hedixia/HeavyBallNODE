import torch

x = torch.range(1, 20)
print(x)

print(x.unfold(0, 5, 2))