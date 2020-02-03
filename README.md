# mAP_3Dvolume

Introduction:
This repo contains a tool to evaluate the mean average precision score (mAP) of 3D segmentation volumes. Our tool relates on this fork of the COCO API: https://github.com/youtubevos/cocoapi

Requirements:
You can use the .yml file to install the requirements with the following command:


The master branch is running with python 2.7 but can easily be adapted to run with python 3 if needed. Please note that in python 3 you need to convert the 'counts' key of the mask.encode() output from bytes to string in order to avoid the issue described here:
https://github.com/cocodataset/cocoapi/issues/70


How it works:
1) Load the following 3D arrays:
a. GT segmentation volume
b. prediction segmentation volume
c. affinity matrix / sigmoid output matrix in order to get the prediciton score of each voxel

2) Create the necessary json files for the COCO API

3) Evaluate the model performance with mAP by using the COCO API fork of youtubevos
