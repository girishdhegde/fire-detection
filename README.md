# fire-detection
***

Fire and Smoke Detection is finding exactly where is the fire and smoke in image by putting bounding boxes. To achieve that we thought of various techniques.

*  Image Processsing
*  Learning Based

Each of the above mentioned techniques have their own pros and cons.

## Image Processing:
***
First we thought of simple image processing technique:
1.  Collect some fire images(100).
2.  Label Bounding Boxes for these small dataset.
3.  Plot **Histogram** channelwise(RGB).
4.  Find **Upper Threshold** and **Lower Threshold** for each channel.
5.  Testing
5.  Perform **thresholding** based on these found thresholds.
6.  Apply **Morphological Transforms** like **erosion or dilation** with proper **structural element**.
7.  We have approximate bounding boxes.
8.  Further **Hough Tranform** or other **Contour Hunting** Techniques can be used to get co-ordinates of rectangle.
9.  Here's similar [example](https://stackoverflow.com/a/51756462/14108734)

Main drawback of these image processing techniques is they **dont't generalize** very well. We have **engineere** all features by hand. But they are **very fast**. 
We thought of extracting some features like edges inside the fire i.e. red line like structure and using it for detection. But we din't get any considerable results. So we went for **Deep Learning** Techniques.

## Deep Learning:
***
We know that **Convolution Neural Networks** are very good at extracting **features** and perform very well on **Vision Tasks** because of thier properties like:
1.  taking account of spatially local pattern.
2.  Low parameters due to sparse connections and Parmeter sharing.
3.  Translational Invariance
4.  Capturing Global view from receptive fields

So we went for CNN's. When it comes to **Object Detection** **YOLO** models are best arguably. So we created our own **Fire Smoke detector** based on **YOLO** architectures.

### [Yolo-V1](https://arxiv.org/abs/1506.02640):
***
We have implemented an CNN object detection architecture similar to  yolov1 completely from **Scratch** in python using **PyTorch**. Main Features of Yolo are:
1.  **Output Encoding.**
2.  Non Maximum Suppression.
3.  Intersection Over Union.
4.  **Yolo Loss function.**
5.  Feature Extractor(CNN)

#### Output Encoding
***
*  image is divided into **S × S grid**.
*  Each grid cell can predict **B - Bounding Boxes** and a object **class**(one hot encoded).
*  Bounding box is encoded as **[x, y, w, h, p]** where,
   (x, y) - x and y co-ordinates of the object **centre** w.r.t to that **cell's top-left corner** i.e. in **range(0, 1)**.
   (w, h) - **width and height** of the object w.r.t to that  whole image i.e. in **range(0, 1)**.
   p - **confidence** score that object is there at that boundindg box.
*  If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
*  Each grid cell also predicts **C** conditional class probabilities, **Pr(Classi|Object)**.
*  Hence prection is **(SxSx(C+5B))** tensor
*  In our case **S = 7, C(no. of classes) = 2, B(bbox per cell) = 2**.
![Output Encoding](/screenshots/encoding.png)

#### Intersection Over Union:
***
IOU is **evaluation metric** which can be used to measure accuracy of object detection based on area of bounding box and actual object.
**IOU = Area of Overlap / Total Area**

![IOU](/screenshots/iou.png)


#### Non Maximum Suppression:
***
NMS is simple algorithm to select **right bounding box** among **overlapping** bounding boxes.
1.  Keep bounding box with **Highest Confidence Score**.
2.  **Compare** other overlapping bounding boxes with the above selected box through **IOU** metric.
3.  If IOU is **Less** than some **threshold keep** the box **else discard** the box.
4.  Hence it keeps **most prominent** and **unique** bounding boxes with less overlapping.
5.  Note: compare among the only overlapping bounding boxes which belongs to **same class**.

#### Yolo Loss Function
***
Below image shows well engineered Yolo Loss Function. We implemented this loss function entirely from scratch in PyTorch.
![loss](/screenshots/loss.png)


#### Feature Extractor
***
Feature axtractor is basically **backbone** of Yolo which is a **CNN** which extracts features from given input image. Mainly **Darknet** is used for this purpose in **Yolo**.
**We** implemented both **Darknet** backbone and also **ResNet50** as backbone. In our case we **ResNet50** was working better as it was **pretrained** and we only trained some **last** layers through **Transfer Learning** due to **small dataset constarint**.


### [Yolo-V4](https://arxiv.org/pdf/2004.10934.pdf)
***
We trained **Yolo-V1** entirely from scratch but it was still **lagging** in results section as it is small network and 4-5 years old technology. So then shifted our focus towards **New State of the Art** object detector **Yolo-V4**. Due to **time** contraints we are unable to write whole yolov4 from scratch. So we have used some help from existing yolov4 implementations. We wrote only some dataloader, testing, demo/inference codes from scratch for yolov4. And we got very good results.

## Dataset Generation
***

For fire smoke detection dataset with bounding box annotations is required. But there is no large and good fire-smoke dataset is out there. So we have to create a dataset of our own.

*  Collection of fire-smoke images
     *  Gathered around **2000 images** from already available [**fire-smoke** classification](https://www.kaggle.com/search?q=fire+and+smoke+detection+datasetSize%3Amedium) **Kaggle** datasets.
     *  Downloaded around **500 images** directly from web.
*  Data Cleaning: After removing bad images we left with around total of **1000 images**.
*  Annotation: We have used [**SuperAnnotate**](https://superannotate.com/) tool for **bounding boxes** annotation.
*  Coco Lables: We have converted the annotations from that tool to **Coco Dataset** bbox lable format through som **Python Scripting**.
*  Finally we created a dataset of around **750 fire** bounding boxes and **400 smoke** bounding boxes.

***
## Training
***
Length of Training set = 800 images,
Length of Validation set = 20 images,
Yolo-v1 Trained from scratch,
Yolo-v1(ResNet50) Transfer Learning,
Yolo-v4 Trained from scratch.

Models are trained on **Nvidia GTX 1660-Ti** 6GB Graphics Card(Acer Predator Laptop) using Pytorch.

Inference time on cpu is around 0.5seconds per image. (When Batch size = 1)

***
| Model            | Parameters    | Trainable Params  | Epochs | Permormance |
| -----------------| ------------- | ----------------- |--------|-------------| 
| Yolo-V1          | 112166284     | 112166284 | 50     | Bad
| Yolo-V1(ResNet50)| 26208908      | 17665612  | 50     | Average
| Yolo-V4          | 52921437      | 52921437  | 300    | Best


***
## Results
***
Here are some sample results(Yolo-V4).
| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604"  src="/screenshots/loss.png">  blah |

***
## What Went Wrong
Images
Reasoning
***

***
Conclusion
***


## Scan the QR code Open with Drive to Download the Android App:

![application](/screenshots/app.png)
