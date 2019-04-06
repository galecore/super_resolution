import torch

def psnr(errs):
    return -10 * torch.log(errs) / torch.log(10. * torch.ones(errs.size()))