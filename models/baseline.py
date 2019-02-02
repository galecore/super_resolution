import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size=7, activation=torch.nn.ELU):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_layers, out_layers, kernel_size, padding=kernel_size//2)
        self.activation = activation()
        
        #init weights 
        torch.nn.init.xavier_normal_(self.conv.weight, torch.nn.init.calculate_gain('leaky_relu'))
        
    def forward(self, x):
        out = self.activation(self.conv(x))
        return out     

class UpConvBlock(torch.nn.Module): # with PixelShuffle
    def __init__(self, in_layers, kernel_size=3, activation=torch.nn.ELU, upscale=2):
        super(UpConvBlock, self).__init__()
        self.upconv = torch.nn.Conv2d(in_layers, (upscale**2)*in_layers, kernel_size, padding=kernel_size//2)
        self.pixelshuffle = torch.nn.PixelShuffle(upscale)
        
        #init weights
        torch.nn.init.xavier_normal_(self.upconv.weight, torch.nn.init.calculate_gain('leaky_relu'))
        
    def forward(self, x):
        out = self.upconv(x)
        out = self.pixelshuffle(out)
        return out    


class Baseline(torch.nn.Module):
    def __init__(self, device, *sequence):
        super(Baseline, self).__init__()
        self.model = torch.nn.Sequential(*sequence).to(device)
        self.device = device
        
    def forward(self, x):
        x = self.model.forward(x.to(self.device))
        return x

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = self.model.forward(x.to(self.device))
        self.model.train()
        return x.detach().cpu()
