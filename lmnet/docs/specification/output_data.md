# Output Data Specification

Define inference (predict) output data format specification for each tasks. 

* [Classification](#imageclassification)
* [Object detection](#imageobject_detection)
* [Semantic segmentation](#imagesemantic_segmentation)

## Type of inference output data
There are two type of inference output data.

### Tensor (np.array)   
`Tensor` type output is post processed  multi dimension array.  
The output don't include meta data. When you handle this type output for application, you need more information such as class names, the input image size before resizing.  

### json
`json` type output include meta-data and well done for application.  
For example, predicted bounding boxes already resized to fit raw input image size.  



## Tasks
Currently we have only 3 image tasks. 
I will add more image tasks and audio and more.

* Image
  * Classification: `IMAGE.CLASSIFICATION`
  * Object Detection: `IMAGE.OBJECT_DETECTION`
  * Semantic Segmentation: `IMAGE.SEMANTIC_SEGMENTATION`


-----

# Format Specification
Define output data format.

## `IMAGE.CLASSIFICATION`
### Tensor   
Output shape is `[batch size, number of class]` .  
In a image, each value is in the range (0, 1), and all the value add up to 1. 

### json
"results" include batch size prediction result.

```
{
    "classes": [
        {
            "id": 0,
            "name": "airplane"
        },
        {
            "id": 1,
            "name": "automobile"
        },
        ...
    ],
    "date": "2018-06-08T11:43:38.759582+00:00",
    "results": [
        {
            "file_path": "/home/lmnet/tests/fixtures/datasets/lm_things_on_a_table/Data_multi/30298aabba35f9bde39cf4a08af11bd9.jpg",
            "prediction": [
                {
                    "class": {
                        "id": 0,
                        "name": "airplane"
                    },
                    "probability": "0.08696663"
                },
                {
                    "class": {
                        "id": 1,
                        "name": "automobile"
                    },
                    "probability": "0.049272012"
                },
                ...
            ]
        },
        {
            "file_path": "/home/lmnet/tests/fixtures/datasets/lm_things_on_a_table/Data_multi/30298aabba35f9bde39cf4a08af11bd9.jpg",
            "prediction": [
                {
                    "class": {
                        "id": 0,
                        "name": "airplane"
                    },
                    "probability": "0.08696663"
                },
                {
                    "class": {
                        "id": 1,
                        "name": "automobile"
                    },
                    "probability": "0.049272012"
                },
                ...
            ]
        }
    ],
    "task": "IMAGE.CLASSIFICATION",
    "version": 0.2
}
```


## `IMAGE.OBJECT_DETECTION`

### Tensor   
Shape is `[batch size, number of predict boxes, 6(x(left), y(top), w, h, class_id, score)]`.   
For example `YOLOv2` network, The post process `FormatYOLOV2` convert bare convolution network output to the shape.
Then, all other post process for object detection (ie. NMS) assume input as the shape and output the shape.


### Json
"results" include batch size prediction result.  
"box" order is `(x(left), y(top), width, height)` , the size is fitted to raw input image size.


```
{
    "classes": [
        {
            "id": 0,
            "name": "hand"
        },
        {
            "id": 1,
            "name": "salad"
        },
        {
            "id": 2,
            "name": "steak"
        },
        ....
    ],
    "date": "2018-06-08T12:07:41.531325+00:00",
    "results": [
        {
            "file_path": "/home/lmnet/tests/fixtures/datasets/lm_things_on_a_table/Data_multi/e2c58de56674f95d2edd4c3b4de7c39f.jpg",
            "prediction": [
                {
                    "box": [
                        299.9669580459595,
                        155.85418617725372,
                        23.924002647399902,
                        24.240950345993042
                    ],
                    "class": {
                        "id": 3,
                        "name": "whiskey"
                    },
                    "score": "0.10005419701337814"
                },
                {
                    "box": [
                        443.9669580459595,
                        443.8541861772537,
                        23.924002647399902,
                        24.240950345993042
                    ],
                    "class": {
                        "id": 3,
                        "name": "whiskey"
                    },
                    "score": "0.10005419701337814"
                }
            ]
        },
        {
            "file_path": "/home/lmnet/tests/fixtures/datasets/lm_things_on_a_table/Data_multi/a3e53b8217b60e7754b8031f929034f3.jpg",
            "prediction": [
                {
                    "box": [
                        299.9669580459595,
                        155.85418617725372,
                        23.924002647399902,
                        24.240950345993042
                    ],
                    "class": {
                        "id": 3,
                        "name": "whiskey"
                    },
                    "score": "0.10005419701337814"
                },
                {
                    "box": [
                        443.9669580459595,
                        443.8541861772537,
                        23.924002647399902,
                        24.240950345993042
                    ],
                    "class": {
                        "id": 3,
                        "name": "whiskey"
                    },
                    "score": "0.10005419701337814"
                }
            ]
        },
    ],
    "task": "IMAGE.OBJECT_DETECTION",
    "version": 0.2
}
```




## IMAGE.SEMANTIC_SEGMENTATION


### Tensor   
Shape is `[batch size, height, width, number of class]`.
The height and width is network input image size.
In a image, each value is in the range (0, 1). In a pixel, all value add up to 1. 


### Json
"results" include batch size prediction result.  
"prediction" include each class result.  
"mask" is base64 encoded gray-scale PNG image, the range (0, 255) that is scaled (0, 1) by 255.
the size is fitted to raw input image size.

```
{
    "classes": [
        {
            "id": 0,
            "name": "sky"
        },
        {
            "id": 1,
            "name": "building"
        },
        ...
    "date": "2018-06-08T12:33:33.681521+00:00",
    "results": [
        {
            "file_path": "/home/lmnet/tests/fixtures/camvid/0001TP_006870.png",
            "prediction": [
                {
                    "class": {
                        "id": 0,
                        "name": "sky"
                    },
                    "mask": "iVBORw0KGgoAAAANSUhEUgAAAeAAAAFoCAAAAACqXHf8AAAlhElEQVR4nO2deWCcVbnwfzOZ7Jkkk31PG9Is3dNSSlsKlCKlLAoKCCgo6MUFXBC9blf9rgvq5yfqvaJXVC6owEWwQsuVUqGF0qaFLknXpGnTptn3fZns3x/ZJ7O8885531l6fn/Nct73PMkz57znPOdZDEV0ZvM+WQ1WBsHIGIQPYEygGeLbgHTqIunDNRY6Z1"
               },
               ...
            ]
        }
    ],
    "task": "IMAGE.SEMANTIC_SEGMENTATION",
    "version": 0.2
}
