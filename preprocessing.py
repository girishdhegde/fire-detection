import os
import numpy as np
import cv2
# import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import pickle
import math

def preprocess(img_path, lbl_path, out_img_path, out_lbl_path, size=448):
    cnt = -1
    for i in range(1, 334):
        img = cv2.imread(img_path+str(i)+'.jpeg')
        m, n, _ = img.shape
        scale_m, scale_n = (size / m, size / n)
        scaled_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        with open(lbl_path+str(i)+'.txt') as f:
            obj = f.readlines()
        clas = int(obj[0]) - 1
        if clas < 2:
            print(cnt)
            cnt += 1
            boxes = []
            for box in obj[1:]:
                x1, y1, x2, y2 = list(map(int, box.split()))
                x1 = int(x1 * scale_n)
                y1 = int(y1 * scale_m)
                x2 = int(x2 * scale_n)
                y2 = int(y2 * scale_m)
                w  = x2 - x1
                h  = y2 - y1
                x  = int(x1 + w / 2)
                y  = int(y1 + h / 2)
                boxes.append([x1, y1, x2, y2, x, y, w, h, clas])
            cv2.imwrite(out_img_path+str(cnt)+'.png', scaled_img)
            with open(out_lbl_path+str(cnt)+'.json', 'w') as f:
                json.dump(boxes, f)

            # print(clas, boxes)
            # img = cv2.rectangle(scaled_img, (x1, y1), (x2, y2), color=(255, 0, 0))
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # break


def encodeYolo(path, path2, size=448, S=7, C=2, total=322):
    cell_size = size / S
    for i in range(total):
        print(i)
        with open(path+str(i)+'.json') as f:
            lbl = json.load(f)
        target = [[[0.0 for _ in range(5+C)] for _ in range(S)] for _ in range(S)]
        for box in lbl:
            x, y, w, h, c = box[4:]
            cell_x, center_x = divmod(x-1, cell_size) 
            cell_y, center_y = divmod(y-1, cell_size) 
            w /= size
            h /= size
            x = center_x / cell_size
            y = center_y / cell_size
            target[math.floor(cell_y)][math.floor(cell_x)][:5] = [x, y, w, h, 1.0]
            # one hot encoding
            target[math.floor(cell_y)][math.floor(cell_x)][5+c] = 1.0
        with open(path2+str(i)+'.json', 'w') as f:
            json.dump(target, f, indent=4)


if __name__ == '__main__':

    # # PREPROCESSING DATA

    # img_path = './data/Images/'
    # lbl_path = './data/Labels/'
    # out_img_path = './trainset/images/'
    # out_lbl_path = './trainset/labels/'

    # size = 448

    # preprocess(img_path, lbl_path, out_img_path, out_lbl_path, size)

    # encodeYolo('./trainset/labels/', './trainset/elabels/', size=448, S=7, C=2, total=322)
    
    from utils import convert

    for idx in range(300):
        img = cv2.imread(f'./trainset/images/{idx}.png')
        with open(f'./trainset/elabels/{idx}.json') as f:
            lbl = json.load(f)
        lbl = torch.tensor([lbl])
        # print(lbl.shape)

        pred = convert(lbl, s=7, B=1)[0].numpy()
        # print(pred.shape)

        for row in range(7):
            for col in range(7):
                if pred[row, col, 4] > 0.0:
                    pt1 = (pred[row, col,  : 2] * 448).astype(np.int32)
                    pt2 = (pred[row, col, 2: 4] * 448).astype(np.int32)
                    img = cv2.rectangle(img, tuple(pt1), tuple(pt2), color=(0, 0, 255))
                    img = cv2.putText(img, str(pred[row, col, 5])+','+str(pred[row, col, 6]), 
                                      tuple(pt1), fontFace=0, fontScale=.5, color=(0, 255, 255))
        
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


