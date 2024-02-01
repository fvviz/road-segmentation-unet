import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.conv(X)
    

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs =nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels= feature, kernel_size=2, stride=2)) # multiply feature by 2 to account for skip connection 
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)
        self.final=  nn.Conv2d(features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, X):
        skip_connections = []
        for i, down in enumerate(self.downs):
            X = down(X)
            skip_connections.append(X)
            X = self.pool(X)

        X = self.bottleneck(X)
        skip_connections = list(reversed(skip_connections))

        for i in range(0, len(self.ups), 2):
            X = self.ups[i](X)
            skip_conn = skip_connections[i//2]
            concat_skip = torch.cat((skip_conn,X), dim=1)
            X = self.ups[i+1](concat_skip)

        return self.final(X)