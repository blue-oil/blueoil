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
