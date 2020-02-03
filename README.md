# mAP_3Dvolume

## Introduction:
This repo contains a tool to evaluate the mean average precision score (mAP) of 3D segmentation volumes. 

Our tool relates on this fork of the COCO API: https://github.com/youtubevos/cocoapi

## Requirements:
- You can use one of the following two commands to install the required packages:
```
conda install --yes --file requirements.txt
pip install requirements.txt
```


- Then, go through https://github.com/youtubevos/cocoapi to install the pycocotools:
```
git clone https://github.com/youtubevos/cocoapi.git
cd cocoapi/PythonAPI
# To compile and install locally 
python setup.py build_ext --inplace
# To install library to Python site-packages 
python setup.py build_ext install
```


- The master branch is running with python 2.7 but can easily be adapted to run with python 3 if needed. Please note that in python 3 you need to convert the 'counts' key of the mask.encode() output from bytes to string in order to avoid the issue described here:
https://github.com/cocodataset/cocoapi/issues/70


## How it works:
1) Load the following 3D arrays:
- GT segmentation volume
- prediction segmentation volume
- affinity matrix / sigmoid output matrix in order to get the prediciton score of each voxel

2) Create the necessary json files for the COCO API

3) Evaluate the model performance with mAP by using the COCO API fork of youtubevos
