import numpy as np
import cv2
from utils import convert


def viz(img, pred, c=20, classes=None, colors=None):
    bs, S, _, t = pred.shape
    b = (t - c) // 5
    size = img.shape[1]

    predi = convert(pred, s=S, B=b)[0].numpy()
    # print(predi)
    img = np.array(img[0]).astype(np.uint8)
    for row in range(S):
        for col in range(S):
            for boxi in range(1, b+1):
                if predi[row, col, 5*boxi-1] > 0.5:
                    pt1 = (predi[row, col,  5*(boxi-1): 5*(boxi-1)+2] * size).astype(np.int32)
                    pt2 = (predi[row, col, 5*(boxi-1)+2: 5*(boxi-1)+4] * size).astype(np.int32)
                    x, y = pt1
                    if (x > 0) and (x < size) and (y > 0) and (y < size):
                        x, y = pt2
                        if (x > 0) and (x < size) and (y > 0) and (y < size):
                            if (pt1[0] != pt2[0]) and (pt1[1] != pt2[1]):
                                # print(pt1, pt2)
                                # print('yes')
                                img = cv2.rectangle(img, tuple(pt1), tuple(pt2), color=(255, 0, 0), thickness=3)
                    # img = cv2.putText(img, str(predi[row, col, 5*b])+','+str(pred[row, col, 5*b+1]), 
                    #                     tuple(pt1), fontFace=0, fontScale=.5, color=(0, 255, 255))
    # print(img.shape)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return cv2.UMat.get(img)
    return img

def viz_class(img, pred, classes=None, colors=None):
    size = img.shape[1]
    for clas in pred:
        for boxes in clas:
            for box in boxes:
                # print(box.shape)
                box = box.squeeze()
                pt1 = (int(box[0] * size), int(box[1] * size))
                pt2 = (int(box[2] * size), int(box[3] * size))
                # print(pt1.shape, pt2.shape)
                x, y = pt1
                if (x > 0) and (x < size) and (y > 0) and (y < size):
                    x, y = pt2
                    if (x > 0) and (x < size) and (y > 0) and (y < size):
                        if (pt1[0] != pt2[0]) and (pt1[1] != pt2[1]):
                            img = cv2.rectangle(img, tuple(pt1), tuple(pt2), color=(0, 0, 255), thickness=3)
                        # img = cv2.putText(img, str(predi[row, col, 5*b])+','+str(pred[row, col, 5*b+1]), 
                        #                     tuple(pt1), fontFace=0, fontScale=.5, color=(0, 255, 255))
    # print(img.shape)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

