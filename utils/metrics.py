# from math import log 
# def psnr(mse):
#     return 20*log(255, 10) - 5*log(mse, 10)
from math import log 
def psnr(mse):
    return -10 * log(mse, 10)