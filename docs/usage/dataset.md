# Prepare training dataset

It is necessary to prepare the training dataset before you start blueoil.

The dataset consists of each example that is pair consisting of an input image data and a desired output data (the supervisory signal). In addition to that, class (category) names etc.
The information of the desired output data changes according to the target task (classification, object detection). In response to the desired task, blueoil supports several public dataset formats and the format output by [Delta-Mark](https://delta.leapmind.io/mark/) service.

## Dataset format

### training vs validation
With all dataset formats, you can select the root path of data for training and validation.
The path of training data is required, validation is optional. When you only select training data path, validation data are created from training data automatically. In the following description, `training_dataset_path` is the root path of training data and` validation_dataset_path` is the root path of validation data.

### List of supported dataset format

- Task type: `Classification`
  - [Caltech 101](#caltech-101)
  - [DeLTA-Mark classification](#delta-mark-classification)

- Task type: `Object Detection`
  - [OpenImagev4](#openimagev4)
  - [DeLTA-Mark object detection](#delta-mark-object-detection )

- Task type: `Semantic Segmentation`
  - [CamvidCustom](#camvidcustom)

### Caltech 101
As following [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/), a dataset format where category names are the names of the subdirectories.
The subdirectory under `training_dataset_path` or` validation_dataset_path` becomes the class name and images are located under the subdirectory.

Example

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


# If you set `validation_dataset_path`, you can locate images as the same manner.

validation_dataset_path
├── class_0
│   ├── 0002.jpg
│
├── class_1
│   ├── 1234.jpeg

```

### DeLTA-Mark classification
You can download data of this format by using [Delta-Mark](https://delta.leapmind.io/mark/) service.


### OpenImagev4

It is data format based on [Open Images Dataset V4](https://storage.googleapis.com/openimages/web/index.html).

Place the following files and directory under `training_dataset_path` or` validation_dataset_path`.
* `annotations-bbox.csv`: The CSV that each row defines one bounding box. The field specifications are described in `Boxes` section of [Open Images Dataset V4 of Data Formats](https://storage.googleapis.com/openimages/web/download.html#dataformats).
* `class-descriptions.csv`: The CSV each row defines class name. It is not necessary for validation data. 
 The field specifications are described in `Class Names` section of [Open Images Dataset V4 of Data Formats](https://storage.googleapis.com/openimages/web/download.html#dataformats).
* `images`: All images are located under the directory.


Example

```
training_dataset_path
├── annotations-bbox.csv
├── class-descriptions.csv
└── images
    ├── 000002b66c9c498e.jpg
    ├── 000002b97e5471a0.jpg
    ├── 000002c707c9895e.jpg


# If you set `validation_dataset_path`, you can locate images as the same manner.
validation_dataset_path
├── annotations-bbox.csv
└── images
    ├── 0001eeaf4aed83f9.jpg
    ├── 000595fe6fee6369.jpg
    ├── 00075905539074f2.jpg
```


Example of `annotations-bbox.csv`.

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


Example of `class-descriptions.csv`
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



### DeLTA-Mark object detection
You can download data of this format by using [Delta-Mark](https://delta.leapmind.io/mark/) service.


### CamvidCustom

It is data format based on [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). We can see sample dataset from [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial) repository. In CamvidCustom dataset format, training and annotation data are both binary image file.

Place the following files and directory.

- `label_display_colors.txt`: Class information file like under the example.
- `train.txt`: Pair of data and annotation image file paths. see below example.
- `train`: All training images are located under the directory.
- `trainannot`: All training annotation images are located under the directory.
- `val.txt`: Pair of data and annotation image file paths. see below example.
- `val`: All test images are located under the directory.
- `valannot`: All test annotation images are located under the directory.

Example of dataset structure

```
training_dataset_path
├── label_colors.txt
├── train.txt
├── train
│   ├── 0001TP_006690.png
│   ├── 0001TP_006720.png
│   ├── 0001TP_006750.png
└── trainannot
    ├── 0001TP_006690.png
    ├── 0001TP_006720.png
    ├── 0001TP_006750.png

# If you set `validation_dataset_path`, you can locate files as the same manner.
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

Example of `train.txt`

```
train/0001TP_006690.png trainannot/0001TP_006690.png
train/0001TP_006720.png trainannot/0001TP_006720.png
train/0001TP_006750.png trainannot/0001TP_006750.png
train/0001TP_006780.png trainannot/0001TP_006780.png
train/0001TP_006810.png trainannot/0001TP_006810.png
```

Example of training image is [here](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/CamVid/train/0001TP_006690.png), and annotation image is [here](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/CamVid/testannot/0001TP_008550.png). Annotations are gray scale image.

Example of `val.txt`

```
val/0016E5_07959.png valannot/0016E5_07959.png
val/0016E5_07961.png valannot/0016E5_07961.png
val/0016E5_07963.png valannot/0016E5_07963.png
val/0016E5_07965.png valannot/0016E5_07965.png
val/0016E5_07967.png valannot/0016E5_07967.png
```

Example of `label_display_colors.txt`

```
128 128 128	Sky
128 0 0		Building
192 192 128	Column_Pole
128 64 128	Road
...
0 0 0		Ignore
```

If last class name is `Ignore`,  `Ignore` class will not be used for training.
You can use `Ignore` class for Background (unlabelled) class.