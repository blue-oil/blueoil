# Prepare training dataset

It is necessary to prepare your training dataset before you start blueoil.

A dataset is a set of pairs consisting of an input image and a desired output (the supervisory signal, e.g. class (category) names).
The type of the desired output data changes according to the target task (classification, object detection or segmentation). In response to the desired task, blueoil supports several public dataset formats.

## Dataset format

### training vs validation
With all dataset formats, you can choose the root path of the data to be used for training and validation.
Training data is required; validation data is optional. When you only provide training data, the validation dataset is created from the training data automatically. In the following description, `training_dataset_path` is the root path of the training data and `validation_dataset_path` is the root path of the validation data.

### List of supported dataset formats

- Task type: `Classification`
  - [Caltech 101](#caltech-101)

- Task type: `Object Detection`
  - [OpenImagev4](#openimagev4)

- Task type: `Semantic Segmentation`
  - [CamvidCustom](#camvidcustom)

- Task type: `Keypoint Detection`
  - [MSCOCO_2017 keypoint detection](#mscoco-2017-keypoint-detection)

### Caltech 101
[Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/), a dataset format where category names are the names of the subdirectories.
The subdirectory names under `training_dataset_path` and `validation_dataset_path` represent the class names and images are located under the those directories.

Example:

```
training_dataset_path
├── class_0
│   ├── 0001.jpg
│   ├── xxxa.jpeg
│   ├── yyyb.png
│
├── class_1
│   ├── 123.jpg
│   ├── xxxa.jpeg
│   ├── wwww.jpg


# If you set `validation_dataset_path`, you can locate images in the same manner.

validation_dataset_path
├── class_0
│   ├── 0002.jpg
│
├── class_1
│   ├── 1234.jpeg

```


### OpenImagev4

[Open Images Dataset V4](https://storage.googleapis.com/openimages/web/index.html)

Place the following files and directories under `training_dataset_path` (and `validation_dataset_path`).
* `annotations-bbox.csv`: a CSV file where each row defines one bounding box. The field specifications are described in `Boxes` section of [Open Images Dataset V4 of Data Formats](https://storage.googleapis.com/openimages/web/download.html#dataformats).
* `class-descriptions.csv`: a CSV file where each row defines a class name. It is not necessary for validation data.
 The field specifications are described in the `Class Names` section of [Open Images Dataset V4 of Data Formats](https://storage.googleapis.com/openimages/web/download.html#dataformats).
* `images`: all images are located in this directory.


Example:

```
training_dataset_path
├── annotations-bbox.csv
├── class-descriptions.csv
└── images
    ├── 000002b66c9c498e.jpg
    ├── 000002b97e5471a0.jpg
    ├── 000002c707c9895e.jpg


# If you set `validation_dataset_path`, you can locate images in the same manner.
validation_dataset_path
├── annotations-bbox.csv
└── images
    ├── 0001eeaf4aed83f9.jpg
    ├── 000595fe6fee6369.jpg
    ├── 00075905539074f2.jpg
```


Example of `annotations-bbox.csv`:

```
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
000026e7ee790996,freeform,/m/07j7r,1,0.071905,0.145346,0.206591,0.391306,0,1,1,0,0
000026e7ee790996,freeform,/m/07j7r,1,0.439756,0.572466,0.264153,0.435122,0,1,1,0,0
000026e7ee790996,freeform,/m/07j7r,1,0.668455,1.000000,0.000000,0.552825,0,1,1,0,0
000062a39995e348,freeform,/m/015p6,1,0.205719,0.849912,0.154144,1.000000,0,0,0,0,0
000062a39995e348,freeform,/m/05s2s,1,0.137133,0.377634,0.000000,0.884185,1,1,0,0,0
0000c64e1253d68f,freeform,/m/07yv9,1,0.000000,0.973850,0.000000,0.043342,0,1,1,0,0
0000c64e1253d68f,freeform,/m/0k4j,1,0.000000,0.513534,0.321356,0.689661,0,1,0,0,0
0000c64e1253d68f,freeform,/m/0k4j,1,0.016515,0.268228,0.299368,0.462906,1,0,0,0,0
0000c64e1253d68f,freeform,/m/0k4j,1,0.481498,0.904376,0.232029,0.489017,1,0,0,0,0
...
```


Example of `class-descriptions.csv`:
```
...
/m/0pc9,Alphorn
/m/0pckp,Robin
/m/0pcm_,Larch
/m/0pcq81q,Soccer player
/m/0pcr,Alpaca
/m/0pcvyk2,Nem
/m/0pd7,Army
/m/0pdnd2t,Bengal clockvine
/m/0pdnpc9,Bushwacker
/m/0pdnsdx,Enduro
/m/0pdnymj,Gekkonidae
...
```


### CamvidCustom

[CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

We can see a sample dataset in the [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial) repository. In the CamvidCustom dataset format, training and annotation data are both binary image files.

Create the following files and directories.

- `labels.txt`: Class information file.
- `train.txt`: Pairs of data and annotation image file paths. (Training.)
- `train`: All training images are located under this directory.
- `trainannot`: All training annotation images are located under this directory.
- `val.txt`: Pairs of data and annotation image file paths. (Validation.)
- `val`: All test images are located under this directory.
- `valannot`: All test annotation images are located under this directory.

Example of dataset structure:

```
training_dataset_path
├── labels.txt
├── train.txt
├── train
│   ├── 0001TP_006690.png
│   ├── 0001TP_006720.png
│   ├── 0001TP_006750.png
└── trainannot
    ├── 0001TP_006690.png
    ├── 0001TP_006720.png
    ├── 0001TP_006750.png

# If you set `validation_dataset_path`, you can locate the files in the same manner.
validation_dataset_path
├── val.txt
├── val
│   ├── 0016E5_07959.png
│   ├── 0016E5_07961.png
│   ├── 0016E5_07963.png
└── valannot
    ├── 0016E5_07959.png
    ├── 0016E5_07961.png
    ├── 0016E5_07963.png
```

Example of `train.txt`:

```
train/0001TP_006690.png trainannot/0001TP_006690.png
train/0001TP_006720.png trainannot/0001TP_006720.png
train/0001TP_006750.png trainannot/0001TP_006750.png
train/0001TP_006780.png trainannot/0001TP_006780.png
train/0001TP_006810.png trainannot/0001TP_006810.png
```

An example of a training image is [here](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/CamVid/train/0001TP_006690.png), and an annotation image is [here](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/CamVid/testannot/0001TP_008550.png). Annotations are gray scale images.

Example of `val.txt`:

```
val/0016E5_07959.png valannot/0016E5_07959.png
val/0016E5_07961.png valannot/0016E5_07961.png
val/0016E5_07963.png valannot/0016E5_07963.png
val/0016E5_07965.png valannot/0016E5_07965.png
val/0016E5_07967.png valannot/0016E5_07967.png
```

Example of `labels.txt`:

```
Sky
Building
Column_Pole
Road
Ignore
```

`labels.txt` contains a list of classes. Each class corresponds to an annotation image color value.
If the `Ignore` class exists, the corresponding class will not be used for training.


### MSCOCO_2017 keypoint detection

Download [MSCOCO 2017 dataset](http://cocodataset.org/#download).
Specifically, you need to download `2017 Train images`, `2017 Val images` and `2017 Train/Val annotations`.
Then, unzip them and organize dataset files as below. `MSCOCO_2017` should be in your `$DATA_DIR`.

Here is an official [example](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb) about how to use `pycocotools` to load dataset.


```
MSCOCO_2017
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017
│   ├── 000000000009.jpg
├── val2017
│   ├── 000000481404.jpg
```
