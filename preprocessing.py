import os
import numpy as np
import cv2
# import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import pickle
import math
import shutil

def encodeYolo(img_source, lbl_source, img_out, lbl_out, S=7, C=2):
    '''
    Function to convert Coco Labels to Yolo Output Format
    '''
    images = os.listdir(img_source)
    labels = os.listdir(lbl_source)
    cell_size = 1 / S
    for i, img in enumerate(images):
        lbl = img.split('.')[0] + '.txt'
        if lbl in labels:
            shutil.copy(os.path.join(img_source, img), os.path.join(img_out, img))
            target = [[[0.0 for _ in range(5+C)] for _ in range(S)] for _ in range(S)]
            with open(os.path.join(lbl_source, lbl)) as f:
                boxes = f.readlines()
            for box in boxes:
                c, x, y, w, h = list(map(float, box.split()))
                c = int(c)                    
                cell_x = math.floor(x * S) 
                cell_y = math.floor(y * S) 
                x = x * S - cell_x 
                y = y * S - cell_y
                if x < 0 or x > 1 or y < 0 or y > 1:
                    print(x, y)
                target[cell_y][cell_x][:5] = [x, y, w, h, 1.0]
                # one hot encoding
                target[cell_y][cell_x][5+c] = 1.0
            with open(os.path.join(lbl_out, lbl.replace('.txt', '.json')), 'w') as f:
                json.dump(target, f, indent=4)


if __name__ == '__main__':

    # # PREPROCESSING DATA

    img_path = './coco128/images/'
    lbl_path = './coco128/labels/'
    out_img_path = './trainset/images/'
    out_lbl_path = './trainset/labels/'

    encodeYolo(img_path, lbl_path, out_img_path, out_lbl_path, S=7, C=80)
    
    from utils import convert

    images = os.listdir(img_path)
    for i, img in enumerate(images):
        lbl = img.split('.')[0] + '.json'
        image = cv2.imread(os.path.join(out_img_path, img))
        h, w, *_ = image.shape
        with open(os.path.join(out_lbl_path, lbl)) as f:
            lbl = json.load(f)
        lbl = torch.tensor([lbl])
        # print(lbl.shape)

        pred = convert(lbl, s=7, B=1)[0].numpy()
        # print(pred.shape)

        for row in range(7):
            for col in range(7):
                if pred[row, col, 4] > 0.0:
                    x1, y1, x2, y2 = pred[row, col,  :4]
                    x1 = int(x1 * w)
                    x2 = int(x2 * w)
                    y1 = int(y1 * h)
                    y2 = int(y2 * h)
                    # print(x1, y1, x2, y2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=4)
        
        cv2.imshow('img', image)
        q = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if q in {ord('q')}:
            exit()


