import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F

# Target Grid 
S = 7
# Bounding Boxex per Grid Cell
B = 2
# Number of Classes
# C = 20
C = 80
# Prediction Per Cell Vector Length
E = (C+B*5)

resnet50 = models.resnet50(pretrained=True)

# print(resnet50)

# Freezing starting layers weights for transfer learning
for name, param in resnet50.named_parameters():
    # if name.startswith('layer4') or name.startswith('layer3'):
    if name.startswith('layer4'):
        # Learn layer 4
        param.requires_grad = True
        # print(name)
    else:
        param.requires_grad = False 

class yolo(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet50
        # Fully Connected Layers
        self.backbone.fc = nn.Linear(2048, 1024)
        self.act1 = nn.LeakyReLU(0.1)
        self.drop_out1 = nn.Dropout(0.0)
        self.fc2  = nn.Linear(1024, S*S*E)
        
    def forward(self, x):
        '''
        Params:
        x: input images - BatchSize x Channels x Height x Width Tensors - BS x 3 x 448x 448 Tensors
        Returns:
        Bounding Boxes encoded in Yolo Output Format: BS x S X S (C + 5B) tensors - BS x 7 x 7 x 12
        Box: [(x, y, w, h, confidence) x B, Class Prabilities]
        B: Bounding Boxes per Cell - 2
        '''

        # ResNet50 requires 224x244 images
        x = F.interpolate(x, scale_factor=0.5)
        x = self.backbone(x)
        x = self.act1(x)
        x = self.drop_out1(x)
        x = self.fc2(x)
        # reshape into prediction of shape SxSx(C+Bx5)
        x = x.view(x.shape[0], S, S, E)
        return x

if __name__ == '__main__':
    # unit test
    net = yolo()
    print(net)
    inp = torch.randn((3, 3, 448, 448))
    with torch.no_grad():
        out = net(inp)
    print(out.shape)
    # torch.save(net.state_dict(), 'model.pt', )
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total Parameters: ', total_params)
    print('Trainable Parameters: ', trainable_params)
