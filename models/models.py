import torch
import torch.nn.functional as F
from models import blocks as B
import torchvision


class SR_Generator_x2(torch.nn.Module):
    def __init__(self, n_residuals):
        super(SR_Generator_x2, self).__init__()
        self.n_residuals = n_residuals
        self.upsample = 2

        self.pre_conv = B.ConvBlock(3, 64, kernel_size=9, use_bn=False)
        self.residual_seq = torch.nn.Sequential(*[B.DoubleResidualBlock(64, kernel_size=3, activation=torch.nn.PReLU) for _ in range(n_residuals)])
        self.residual_post_conv = B.ConvBlock(64, 64, kernel_size=3, stride=1, use_bn=True, activation=None)
        
        self.upconv = B.UpConvBlock(64, 64, kernel_size=3, upscale=2, activation=torch.nn.PReLU)
        self.post_conv = B.ConvBlock(64, 3, kernel_size=9, activation=torch.nn.Tanh)
        
        self.train()
        
    def forward(self, x):
        pre_residuals = self.pre_conv(x)
        out = self.residual_seq(pre_residuals)
        out = self.residual_post_conv(out) + pre_residuals
        out = self.upconv(out)
        out = self.post_conv(out)
        return out 

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
        self.train()
        return x.detach().cpu()
    
    
class SR_Generator_x4(torch.nn.Module):
    def __init__(self, n_residuals):
        super(SR_Generator_x4, self).__init__()
        self.n_residuals = n_residuals
        self.upsample = 4

        self.pre_conv = B.ConvBlock(3, 64, kernel_size=9, use_bn=False)
        self.residual_seq = torch.nn.Sequential(
            *[B.DoubleResidualBlock(64, kernel_size=3, activation=torch.nn.PReLU) for _ in range(n_residuals)]
        )
        self.residual_post_conv = B.ConvBlock(64, 64, kernel_size=3, stride=1, use_bn=True, activation=None)
        
        self.upconv1 = B.UpConvBlock(64, 64, kernel_size=3, upscale=2, activation=torch.nn.PReLU)
        self.upconv2 = B.UpConvBlock(64, 64, kernel_size=3, upscale=2, activation=torch.nn.PReLU)
        self.post_conv = B.ConvBlock(64, 3, kernel_size=9, activation=torch.nn.Tanh)

        self.train()
        
    def forward(self, x):
        pre_residuals = self.pre_conv(x)
        out = self.residual_seq(pre_residuals)
        out = self.residual_post_conv(out) + pre_residuals
        out = self.upconv1(out)
        out = self.upconv2(out)
        out = self.post_conv(out)
        return out 

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
        self.train()
        return x.detach().cpu()
    
    
class VGGFeatureExtractor(torch.nn.Module): # expects inputs in range [0, 1]
    def __init__(self, cnn=None, feature_layer=34):
        super(VGGFeatureExtractor, self).__init__()
        if cnn is None:
            cnn = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])
        self.features.eval()
        
    def forward(self, x):
        return self.features(x)
    
    
class SR_Discriminator(torch.nn.Module):
    def __init__(self, input_size=96):
        super(SR_Discriminator, self).__init__()
        self.input_size = input_size
        
        self.pre_conv = B.ConvBlock(3, 64, kernel_size=3, use_bn=False, activation=torch.nn.LeakyReLU)

        self.stride_conv1 = B.ConvBlock(64, 64, kernel_size=3, stride=2, activation=torch.nn.LeakyReLU)
        self.deepen_conv1 = B.ConvBlock(64, 128, kernel_size=3, stride=1, activation=torch.nn.LeakyReLU)

        self.stride_conv2 = B.ConvBlock(128, 128, kernel_size=3, stride=2, activation=torch.nn.LeakyReLU)
        self.deepen_conv2 = B.ConvBlock(128, 256, kernel_size=3, stride=1, activation=torch.nn.LeakyReLU)

        self.stride_conv3 = B.ConvBlock(256, 256, kernel_size=3, stride=2, activation=torch.nn.LeakyReLU)
        self.deepen_conv3 = B.ConvBlock(256, 512, kernel_size=3, stride=1, activation=torch.nn.LeakyReLU)
        
        self.stride_conv4 = B.ConvBlock(512, 512, kernel_size=3, stride=2, activation=torch.nn.LeakyReLU)
        
        self.head = B.DiscriminatorHead(input_size**2 // 16**2)
        
    def forward(self, x):
        out = self.pre_conv(x)
        out = self.deepen_conv1(self.stride_conv1(out))
        out = self.deepen_conv2(self.stride_conv2(out))
        out = self.deepen_conv3(self.stride_conv3(out))
        out = self.stride_conv4(out)
        out = self.head(out)
        return out 

    
class Sequential(torch.nn.Module):
    def __init__(self, *sequence):
        super(Sequential, self).__init__()
        self.model = torch.nn.Sequential(*sequence)
        self.model.train()
        
    def forward(self, x):
        x = self.model.forward(x)
        return x

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = self.model.forward(x)
        self.model.train()
        return x.detach().cpu()