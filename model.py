import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration (DarkNet)
conv_structure = [
    # Convolution
    # (input_channels, output_channels, Kernel_size, stride, padding)
    (3, 64, 7, 2, 3),
    # Max pooling
    (2, 2),

    # Convolution
    (64, 192, 3, 1, 1),
    # Max pooling
    (2, 2),
    
    # Convolution
    (192, 128, 1, 1, 0),
    (128, 256, 3, 1, 1),
    (256, 256, 1, 1, 0),
    (256, 512, 3, 1, 1),
    # Max pooling
    (2, 2),
    
    # Convolution
    (512, 256, 1, 1, 0),
    (256, 512, 3, 1, 1),
    (512, 256, 1, 1, 0),
    (256, 512, 3, 1, 1),    
    (512, 256, 1, 1, 0),
    (256, 512, 3, 1, 1),    
    (512, 256, 1, 1, 0),
    (256, 512, 3, 1, 1),
    (512, 512, 1, 1, 0),
    (512, 1024, 3, 1, 1),
    # Max pooling
    (2, 2),
    
    # Convolution
    (1024, 512, 1, 1, 0),
    (512, 1024, 3, 1, 1),
    (1024, 512, 1, 1, 0),
    (512, 1024, 3, 1, 1),
    (1024, 1024, 3, 1, 1),
    (1024, 1024, 3, 2, 1),
    
    # Convolution
    (1024, 1024, 3, 1, 1),
    (1024, 1024, 3, 1, 1),

]

# Target Grid 
S = 7
# Bounding Boxex per Grid Cell
B = 2
# Number of Classes
# C = 20
C = 80
# Prediction Per Cell Vector Length
E = (C+B*5)


class createConv(nn.Module):
    def __init__(self, params):
        super().__init__()
        '''
        Params: CNN, BatchNorm layer params
        Returns: Activation + BatchNorm + Conv
        '''
        self.conv = nn.Conv2d(*params)
        self.bn   = nn.BatchNorm2d(params[1])
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class yolo(nn.Module):
    def __init__(self):
        super().__init__()
        # Creating BAckbone fron Configuration
        self.conv_layers = [nn.MaxPool2d(*params) if len(params) == 2 
                            else createConv(params)
                            for params in conv_structure]
        self.conv_layers = nn.Sequential(*self.conv_layers)

        # Fully Connected Layers
        # self.fc1  = nn.Linear(1024*S*S, 4096)
        self.fc1  = nn.Linear(1024*S*S, 1024)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.dout1 = nn.Dropout(0.0)
        self.fc2  = nn.Linear(1024, S*S*E)
        # self.fc2  = nn.Linear(512, S*S*E)
        
    def forward(self, x):
        '''
        Params:
        x: input images - BatchSize x Channels x Height x Width Tensors - BS x 3 x 448x 448 Tensors
        Returns:
        Bounding Boxes encoded in Yolo Output Format: BS x S X S (C + 5B) tensors - BS x 7 x 7 x 12
        Box: [(x, y, w, h, confidence) x B, Class Prabilities]
        B: Bounding Boxes per Cell - 2
        '''

        x = self.conv_layers(x)
        # flatten
        x = x.view(x.shape[0], -1)
        x = self.act1(self.fc1(x))
        x = self.dout1(x)
        x = self.fc2(x)
        # reshape into prediction of shape SxSx(C+Bx5)
        x = x.view(x.shape[0], S, S, E)
        return x

if __name__ == '__main__':
    # unit Test
    net = yolo()
    print(net)
    inp = torch.randn((3, 3, 448, 448))
    with torch.no_grad():
        out = net(inp)
    print(out.shape)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total Parameters: ', total_params)
    print('Trainable Parameters: ', trainable_params)
