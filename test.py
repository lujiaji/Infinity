import torch
torch.cuda.set_device(3)
ckpt = torch.load("/data/boxunxu/Infinity/infinity_2b_reg.pth", map_location="cuda")
for k, v in ckpt.items():
    print(k, v.dtype)
    break
