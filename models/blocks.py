import torch
import torch.nn.functional as F

#todo: add swish and id 

class FeatureExtractor(torch.nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)
    

class ConvBlock(torch.nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size=7, stride=1, activation=torch.nn.ELU):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_layers, out_layers, kernel_size, stride=stride, padding=kernel_size // 2)
        self.batch_norm = torch.nn.BatchNorm2d(out_layers)
        self.activation = activation()

        # init weights
        torch.nn.init.orthogonal_(self.conv.weight, torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        out = self.activation(self.batch_norm(self.conv(x)))
        return out
    
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size=7, activation=torch.nn.ELU):
        super(ResidualBlock, self).__init__()
        self.conv = ConvBlock(in_layers, out_layers, kernel_size, activation)
        
    def forward(self, x):
        out = x + self.conv(x)
        return out 
    
    
class UpConvBlock(torch.nn.Module):  # with PixelShuffle
    def __init__(self, in_layers, out_layers, kernel_size=3, upscale=2):
        super(UpConvBlock, self).__init__()
        self.upconv = torch.nn.Conv2d(in_layers, (upscale ** 2) * out_layers, kernel_size, padding=kernel_size // 2)
        self.pixelshuffle = torch.nn.PixelShuffle(upscale)

        # init weights
        torch.nn.init.orthogonal_(self.upconv.weight, torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        out = self.pixelshuffle(self.upconv(x))
        return out


class Generator(torch.nn.Module):
    def __init__(self, n_residuals, upsample):
        super(Generator, self).__init__()
        self.n_residuals = n_residuals
        self.upsample = upsample

        self.pre_conv = ConvBlock(3, 64, 9)

        for i in range(self.n_residuals):
            self.add_module('residual_block_' + str(i+1), ResidualBlock(64, 64, 3))

        self.upconv = UpConvBlock(64, 64, 5)
        self.post_conv = ConvBlock(64, 3, 5)

    def forward(self, x):
        out = self.pre_conv(x)

        residual = out.clone()
        for i in range(self.n_residuals):
            residual = self.__getattr__('residual_block_{}'.format(str(i+1)))(residual)

        out = self.upconv(residual)
        out = self.post_conv(out)
        return out
    

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.pre_conv = ConvBlock(3, 64, 3)

        self.stride_conv1 = ConvBlock(64, 64, 3, stride=2)
        self.deepen_conv1 = ConvBlock(64, 128, 3)

        self.stride_conv2 = ConvBlock(128, 128, 3, stride=2)
        self.deepen_conv2 = ConvBlock(128, 256, 3)

        self.stride_conv3 = ConvBlock(256, 256, 3, stride=2)
        self.deepen_conv3 = ConvBlock(256, 512, 3)

        self.stride_conv4 = ConvBlock(512, 512, 3, stride=2)
        self.deepen_conv4 = ConvBlock(512, 512, 3)
        
    def forward(self, x):
        out = self.pre_conv(x)

        out = self.deepen_conv1(self.stride_conv1(out))
        out = self.deepen_conv2(self.stride_conv2(out))
        out = self.deepen_conv3(self.stride_conv3(out))
        out = self.deepen_conv4(self.stride_conv4(out))

        out = F.sigmoid(F.avg_pool3d(out, out.size()))
        return out
