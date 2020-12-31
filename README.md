# fire-detection

Fire and Smoke Detection is finding exactly where is the fire and smoke in image by putting bounding boxes. To achieve that we thought of various techniques.

*  Image Processsing
*  Learning Based

Each of the above mentioned techniques have their own pros and cons.

## Image Processing:

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

We know that **Convolution Neural Networks** are very good at extracting **features** and perform very well on **Vision Tasks** because of thier properties like:
1.  taking account of spatially local pattern.
2.  Low parameters due to sparse connections and Parmeter sharing.
3.  Translational Invariance
4.  Capturing Global view from receptive fields

So we went for CNN's. When it comes to **Object Detection** **YOLO** models are best arguably. So we created our own **Fire Smoke detector** based on **YOLO** architectures.

### [Yolo-V1](https://arxiv.org/abs/1506.02640):

We have implemented an CNN object detection architecture similar to  yolov1 completely from **Scratch** in python using **PyTorch**. Main Features of Yolo are:
1.  **Output Encoding.**
2.  Non Maximal Suppression.
3.  Intersection Over Union.
4.  **Yolo Loss function.**
5.  Feature Extractor(CNN)

#### Output Encoding
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