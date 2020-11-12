# mAP_3Dvolume

## Introduction:
This repo contains a tool to evaluate the mean average precision score (mAP) of 3D segmentation volumes. This tool uses the cocoapi approach for mAP evaluation. The master branch runs super fast. If you wish to test out the master branch, you can run the [legacy branch](https://github.com/ygCoconut/mAP_3Dvolume/tree/legacy). 

## Important notes:
- The tool supposes you load arrays saved as h5 files. Feel free to change the loadh5 function to load something else.
- The tool assumes that the z-axis is the first axis, then x, then y (i.e. gt.shape = (z, x, y), where z represents the slices of your stack). This should not matter in terms of map score though if you load a 3D array.
- There is a variety of flags that you can use. The most important flags are probably -ph and -ps. Choose -ps if you already computed the scores, otherwise you can use -ph to feed the tool with your output layer heatmap.
- In our model output, each voxel has 3 score/affinity values. For this reason, the average instance score is calculated in a way that might not be compatible with your model output. Feel free to adapt the score function.
- Make sure you have converted your semantic segmentation to instance segmentation. Even if all instances have the same category ID, each instance needs a different ID. One way to do it is with skimage.measure.label() (connected components).
- The main branch is running with python 3.7, the legacy branch is running with python 2.7 

## Requirements:
- You can use one of the following two commands to install the required packages:
```
conda install --yes --file requirements.txt
pip install -r requirements.txt
```


- The master branch is running with python 2.7 as a default, but can easily be adapted to run with python 3 if needed.

## How it works:
Run the following command to use the tool:
```
python demo.py -gt demo_data/lucchi_gt_test.h5 -p demo_data/lucchi_pred_UNet_label_test.h5 -ph demo_data/lucchi_pred_UNet_heatmap_test.h5
```
The following steps will be executed by the script:
1) Load the following 3D arrays:
- GT segmentation volume
- prediction segmentation volume
- model prediction matrix / scores matrix in order to get the prediciton score of each voxel

2) Create the necessary tables to compute the mAP:
- iou_p.txt contains the different prediction ids, the prediction scores, and their matched ground trught (gt) ids. Each prediciton is matched with gt ids from 4 different size ranges (based on number of instance voxels). Each of these ranges contains the matched  gt id, its size and the intersection over union (iou) score. 
- iou_fn.txt contains false negatives, as well as instances that have been matched with a worse iou than another instance.  

3) Evaluate the model performance with mAP by using the 3D optimized evaluation script  and the 2 tables mentioned above.

## Citation
If you find it useful in your project, please cite:

```bibtex
@inproceedings{wei2020mitoem,
  author =       {Donglai Wei, Zudi Lin, Daniel Franco-Barranco, Nils Wendt, Xingyu Liu, Wenjie Yin, Xin Huang, Aarush Gupta, Won-Dong Jang, Xueying Wang, Ignacio Arganda-Carreras, Jeff Lichtman, Hanspeter Pfister},
  title =        {MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation from EM Images},
  booktitle = {International Conference on Medical Image Computing & Computer Assisted Intervention (MICCAI)},
  year =         {2020}
}
```
