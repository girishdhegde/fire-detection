import cv2
import json

import torch
from torch.utils.data import DataLoader, Dataset

class dataset(Dataset):
    def __init__(self, img_path, lbl_path, ln=322, device=torch.device('cpu')):
        '''
        img_path: image dataset folder path
        lbl_path: label dataset folder path
        ln: Total images
        '''

        self.img_path = img_path
        self.lbl_path = lbl_path
        self.len = ln
        # self.lbl = []
        # for i in range(ln):
        #     with open(lbl_path+str(i)+'.json') as f:
        #         lbl = json.load(f)
        #     self.lbl.append(lbl)
        # self.lbl = torch.tensor(self.lbl, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        ''' 
        idx: index 0 to self.len
        returns: img and target for training
        '''
        img = cv2.imread(self.img_path+str(idx)+'.png')
        img = torch.tensor(img, dtype=torch.float32, device=self.device)
        with open(self.lbl_path+str(idx)+'.json') as f:
            lbl = json.load(f)
        # target = self.lbl[idx]
        target = torch.tensor(lbl, dtype=torch.float32, device=self.device)
        # [h, w, ch] -> [ch, h, w]
        return img.permute((2, 0, 1)), target

if __name__ == '__main__':
    # unit Test
    img_path = './trainset/images/'
    lbl_path = './trainset/elabels/'
    device = torch.device('cuda')

    dset = dataset(img_path, lbl_path, 322, device)

    dset = next(iter(dset))
    print(dset[0].shape, dset[1].shape)