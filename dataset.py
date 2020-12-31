import cv2
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class dataset(Dataset):
    def __init__(self, img_path, lbl_path, size=(448, 448), device=torch.device('cpu')):
        '''
        img_path: image dataset folder path
        lbl_path: label dataset folder path
        '''
        self.img_path = []
        self.lbl_path = []
        for img in os.listdir(img_path):
            self.img_path.append(os.path.join(img_path, img))
            self.lbl_path.append(os.path.join(lbl_path, img.split('.')[0]+'.json'))
        self.len = len(os.listdir(img_path))
        self.size = size
        self.device = device

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        ''' 
        idx: index 0 to self.len
        returns: img and target for training
        '''
        img = cv2.imread(self.img_path[idx])
        img = torch.tensor(img, dtype=torch.float32, device=self.device)
        with open(self.lbl_path[idx]) as f:
            lbl = json.load(f)
        target = torch.tensor(lbl, dtype=torch.float32, device=self.device)
        # [h, w, ch] -> [ch, h, w]
        img = img.permute((2, 0, 1)).unsqueeze(0)
        img = F.interpolate(img, size=self.size, mode='bilinear')
        return img.squeeze(), target

if __name__ == '__main__':
    # unit Test
    img_path = './trainset/images/'
    lbl_path = './trainset/labels/'
    device = torch.device('cuda')

    dset = dataset(img_path, lbl_path, (448, 448), device)

    dset = next(iter(dset))
    print(dset[0].shape, dset[1].shape)
