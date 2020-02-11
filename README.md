# mAP_3Dvolume

## Introduction:
This repo contains a tool to evaluate the mean average precision score (mAP) of 3D segmentation volumes. 

Our tool relates on this fork of the COCO API: https://github.com/youtubevos/cocoapi

## Important notes:
- The tool supposes you load arrays saved as h5 files. Feel free to change the loadh5 function to load something else.
- The tool assumes that the z-axis is the first axis, then x then y (i.e. gt.shape = (z, x, y), where z represents the slices of your stack). This should not matter in terms of map score though if you load a 3D array.
- In our model output, each voxel has 3 score/affinity values. For this reason, the average instance score is calculated in a way that might not be compatible with your model output. Feel free to adapt the score function.

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
Run the following command to use the tool:
```
python run_eval.py -p "path/to/prediction.h5" -gt "path/to/ground_truth.h5" -ph "path/to/model_output.h5"
```
The following steps will be executed by the script:
1) Load the following 3D arrays:
- GT segmentation volume
- prediction segmentation volume
- affinity matrix / sigmoid output matrix in order to get the prediciton score of each voxel

2) Create the necessary json files for the COCO API

3) Evaluate the model performance with mAP by using the COCO API fork of youtubevos
