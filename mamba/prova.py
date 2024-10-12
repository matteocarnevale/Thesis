import torch 
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)
torch.zeros(1).cuda()