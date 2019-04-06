import torch
import torch.nn.functional as F

#todo: add swish and id     

class ConvBlock(torch.nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size=3, stride=1, use_bn=True, activation=torch.nn.PReLU):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_layers, out_layers, kernel_size, stride=stride, padding=kernel_size // 2)
        
        self.use_bn = use_bn
        if self.use_bn:
            self.batch_norm = torch.nn.BatchNorm2d(out_layers)
            
        if activation:
            self.activation = activation()
        else:
            self.activation = activation
            
        # init weights
        torch.nn.init.xavier_normal_(self.conv.weight, torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.batch_norm(out)
        if self.activation:    
            out = self.activation(out)
        return out
    
    
class ResidualBlock(torch.nn.Module): 
    def __init__(self, in_layers, kernel_size=3, use_bn=True, activation=torch.nn.PReLU):
        super(ResidualBlock, self).__init__()
        self.conv = ConvBlock(in_layers, in_layers, kernel_size=kernel_size, stride=1, use_bn=use_bn, activation=activation)
        
    def forward(self, x):
        out = x + self.conv(x)
        return out 
    
    
class DoubleResidualBlock(torch.nn.Module):
    def __init__(self, in_layers, kernel_size=3, use_bn=True, activation=torch.nn.PReLU):
        super(DoubleResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_layers, in_layers, kernel_size=kernel_size, stride=1, use_bn=use_bn, activation=activation)
        self.conv2 = ConvBlock(in_layers, in_layers, kernel_size=kernel_size, stride=1, use_bn=use_bn, activation=None)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        return x + out 
        
        
class ResidualDenseBlock_5C(torch.nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, in_layers, out_layers, kernel_size=3, use_bn=True, activation=torch.nn.PReLU):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = ConvBlock(in_layers, out_layers, kernel_size, stride=1, use_bn=use_bn, activation=activation)
        self.conv2 = ConvBlock(in_layers + out_layers, out_layers, kernel_size, stride=1, use_bn=use_bn, activation=activation)
        self.conv3 = ConvBlock(in_layers + 2*out_layers, out_layers, kernel_size, stride=1, use_bn=use_bn, activation=activation)
        self.conv4 = ConvBlock(in_layers + 3*out_layers, out_layers, kernel_size, stride=1, use_bn=use_bn, activation=activation)
        self.conv5 = ConvBlock(in_layers + 4*out_layers, out_layers, kernel_size, stride=1, use_bn=use_bn, activation=activation)

    def forward(self, data):
        x1 = self.conv1(data)
        data = torch.cat((data, x1), 1)
        x2 = self.conv2(data)
        data = torch.cat((data, x2), 1)
        x3 = self.conv3(data)
        data = torch.cat((data, x3), 1)
        x4 = self.conv4(data)
        data = torch.cat((data, x4), 1)
        x5 = self.conv5(data)
        return x5.mul(0.2) + x


class RRDB(torch.nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, in_layers, out_layers, kernel_size=3, use_bn=True, activation=torch.nn.PReLU):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(in_layers, out_layers, kernel_size, use_bn=use_bn, activation=activation)
        self.RDB2 = ResidualDenseBlock_5C(in_layers, out_layers, kernel_size, use_bn=use_bn, activation=activation)
        self.RDB3 = ResidualDenseBlock_5C(in_layers, out_layers, kernel_size, use_bn=use_bn, activation=activation)
        
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x

    
class UpConvBlock(torch.nn.Module):  # with PixelShuffle
    def __init__(self, in_layers, out_layers, kernel_size=3, upscale=2, activation=torch.nn.PReLU):
        super(UpConvBlock, self).__init__()
        self.upconv = torch.nn.Conv2d(in_layers, (upscale ** 2) * out_layers, kernel_size, padding=kernel_size // 2)
        self.pixelshuffle = torch.nn.PixelShuffle(upscale)
        
        if activation:
            self.activation = activation()
        else:
            self.activation = activation
            
        # init weights
        torch.nn.init.xavier_normal_(self.upconv.weight, torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        out = self.pixelshuffle(self.upconv(x))
        if self.activation:    
            out = self.activation(out)
        return out

    
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
    
    
class DiscriminatorHead(torch.nn.Module):
    def __init__(self, input_size, linear_units = 1024):
        super(DiscriminatorHead, self).__init__()
        self.flatten = Flatten()
        self.linear1 = torch.nn.Linear(input_size, linear_units)
        self.activation = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(linear_units, 1)
        
    def forward(self, x):
        out = self.flatten(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        return torch.sigmoid(out).flatten()
    