import torch
print(torch.__version__)           #输出Pytorch的版本
print(torch.version.cuda)
print(torch.cuda.is_available())   #看看能否正常调用CUDA
print(torch.cuda.device_count())   # 查看gpu数量
print(torch.cuda.current_device()) # 查看当前gpu号
