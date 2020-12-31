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

Checkout **yoloV4** folder(our testing code) of this repository to test yolov4 on fire-smoke images.

Here's Link To a GitRepo to train Yolo-V4 https://github.com/WongKinYiu/PyTorch_YOLOv4.

## Dataset Generation
***

For fire smoke detection dataset with bounding box annotations is required. But there is no large and good fire-smoke dataset is out there. So we have to create a dataset of our own.

*  Collection of fire-smoke images
     *  Gathered around **2000 images** from already available [**fire-smoke** classification](https://www.kaggle.com/search?q=fire+and+smoke+detection+datasetSize%3Amedium) **Kaggle** datasets.
     *  Downloaded around **500 images** directly from web.
*  Data Cleaning: After removing bad images we left with around total of **1000 images**.
*  Annotation: We have used [**SuperAnnotate**](https://superannotate.com/) tool for **bounding boxes** annotation.
*  Coco Lables: We have converted the annotations from that tool to **Coco Dataset** bbox lable format through som **Python Scripting**.
*  Negative Data: We added about **100 images** as negatives which contains **sun, cloud, etc** images which are not **fire or smoke** but similar.
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
|<img width="1604"  src="/screenshots/3.png">|<img width="1604"  src="/screenshots/8.png">|<img width="1604"  src="/screenshots/67.png">|
|<img width="1604"  src="/screenshots/97.png">|<img width="1604"  src="/screenshots/399.png">|<img width="1604"  src="/screenshots/1274.png">|
***
## What Went Wrong
***
Here are some failure cases:
| | |
|:-------------------------:|:-------------------------:|
|<img width="1604"  src="/screenshots/61.png">|<img width="1604"  src="/screenshots/92.png">|
### Reasoning
The model is unable to detect fire and smoke in some images. Reasons which may be responsible are:
*  Lack of training data: Model trained only on **800 and 400** bounding for boxes fire and smoke class respicively. Atleat **1000** bounding boxes per class is required for proper training.
*  Low Quality Data: Some of the images used are very low quality both resolution and lightning conditions, contrast etc.
*  And for some images like second one shown above we think it won't be possible to detect such kind of white bounding boxes. Because if add more such kind of data model may start to detect **Cloud, Mist, Water, etc** whitish object as smoke.
*  Finally as **fire and smoke** both don't have **definite** shape (unlike the classes in coco dataset) i.e. they are shapeles it will be very difficult for model to learn the features.

## What is it Learning, Conclusion and Future Scope

As mentioned above **fire and smoke** have no **Definite** shape. Still model is able to give very **good** results especially on **fire**.
Here is our **Intuition** behind what may be it learning. 
1.  Most obvious one: Color
2.  It may be learning some **Red Edge** structures inside the fire.
3.  Highly **sharp high frequency boundary regions**.

Future Scope:
*  Labeling More qualitative data
*  Adding adverserial negative data
*  More Data Augmentation
*  Using above mentioned **Intuitions** to come up with some **meta** feature which can be used as a term in **Loss** function for this specific fire-smoke detection.


## Deployment
We have used freely available **Heroku** platform for deploying our model on server. It only provides memory of **512 MB** which is little bit less in perspective of **Deep Learning** Models and no **GPU**(Neccessary for high **speed** inference and low memory **Float16** support). Finally still we are able to run it in Heroku with some **optimization** and bare minimal UI.

As a consequence Our Server can handle only **Single Request** at a time and  takes about **2 - 5** seconds per image inference.
If server shows some error wait 5seconds and re-upload the file. Sorry for the Inconvinience :(.

Here's the link Website:

[**Yolo-V4 Website**](https://firesmokeserver.herokuapp.com/)

[Yolo-V1 Website](https://fire-smoke-app.herokuapp.com/)

or use these links:
1.  https://firesmokeserver.herokuapp.com/
2.  https://fire-smoke-app.herokuapp.com/

**Disclaimer: Image Size Should be Less Than 1MB** Hint: Take ScreenShot(It may reduce the size) and Upload.

We also buit an Android app. Check out the app here
## Scan the QR code Open with Drive to Download the Android App:
| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604"  src="/screenshots/app.png">|<img width="1604"  src="/screenshots/m1.jpg">|<img width="1604"  src="/screenshots/m2.jpg">|

**Or**

Use this link: https://drive.google.com/file/d/1pR8G5bT3LgEfydddjil74tQMfRhIrgoP/view?usp=sharing

***
***
## Here's How to Run The Codes:
***
1.  Clone this repository and open the terminal in the same directory
2.  Install Requirements:
         ```pip install -r requirements.txt``` 
3.  Download or annotate the dataset in [Coco lables format](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)
4.  Give appropriate paths in preprocessing.py and run:
      ```python preprocessing.py```
      ```# Which will conver coco labels to yolo output encoded labels```
5.  Training: Adjust Hyper Parameters and traiset path in train.py and run
      ```python train.py```
6.  For testing/demo
      ```python predict.py [image_path.format] [weight_file.pt]```
      
7. To Run Yolo-V4 Testing Checkout **yoloV4** folder of this repo.
      
 ### Links to Server codes:
 
 1.  [**Yolo-V4** server backend](https://github.com/girishdhegde/fire-server)
 
 2.  [**Yolo-V1** server1 backend](https://github.com/girishdhegde/https://github.com/girishdhegde/firev1_server)
     [**Yolo-V1** server2 backend](https://github.com/girishdhegde/https://github.com/girishdhegde/firev1_server)


## About Us:

1.  GIRISH DATTATRAY HEGDE
    USN: 01FE17BEC054
    gmail: girsihdhegde12499@gmail.com
    phone: 9480626935
    Resume: attached to **resume** folder of this repo.
    StackOverflow: https://stackoverflow.com/users/14108734/girish-dattatray-hegde
    
    
2.  Tushar Pharale
    USN: 01FE17BCS
    gmail:
    phone:
    github: 
