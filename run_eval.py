#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval

import pycocotools.mask as mask
import json


# # How to convert video instance segmentation results into COCO format

# # Main Code

def loadh5py(path, vol="main"):
    return np.array(h5py.File(path, 'r')[vol]).squeeze()


def obtain_id_map(gt, pred):
    ui1,uc1 = np.unique(gt,return_counts=True)
    ui2,uc2 = np.unique(pred,return_counts=True)
    uc1=uc1[ui1>0];ui1=ui1[ui1>0] #ids except background ( =0 )
    uc2=uc2[ui2>0];ui2=ui2[ui2>0]

# create complete mapping of ids for gt and pred:
    # 1. get ids that are not false positives, i.e. gt IDs that have a matching pred pair
    gtids_map = seg_iou3d(gt, pred)[:,:2]
    # 2. get remaining ids, i.e. pred FPs that match with gt background (bg has ID=0)
    for new_pred_id in ui2:
        if np.isin(new_pred_id, gtids_map[:,1].flatten()) == False:
            gtids_map = np.append(np.array(gtids_map), np.array([[0, new_pred_id]]), axis=0)
    return gtids_map


def convert_format(input_videoId, pred_score, pred_catId, pred_segm):
    res_dict = dict()
    res_dict['video_id'] = input_videoId
    res_dict['score'] = pred_score
    res_dict['category_id'] = pred_catId
    
    res_dict['segmentations'] = []
    # move z axis to last dim in order to encode over z; mask.encode needs fortran-order array    
    res_dict['segmentations'] = mask.encode(np.asfortranarray(np.moveaxis(pred_segm, 0, -1)))
    for i in range(len(res_dict['segmentations'])):
        # python2 and python3 bug with bytes and strings to be avoided for "counts"
        res_dict['segmentations'][i]['counts'] = res_dict['segmentations'][i]['counts'].decode('ascii')
        # make z-slices without the specific instance None
        if np.sum(pred_segm[i]) == 0:
            res_dict['segmentations'][i] = None
    return res_dict    


def affinity_score(affinity, mask0, only_z=True):
    if only_z == True:
        return np.mean(affinity[0][mask0.squeeze()].flatten())/255
    else:
        return np.mean( np.mean(affinity, axis=0).squeeze()[mask0.squeeze()].flatten() )

def obtain_masks(gt, pred, gtids_map, only_pred=True):
    """This function takes label prediction and label ground truth of a 3D vol as input and returns the masks
    of the ground truth and the prediction of all instances"""
    mask_gt = []
    mask_pred = [] 
    
    if len(np.array(gtids_map).shape) == 1:
        gtids_map = gtids_map.reshape([1,2])
        
    for i in range(np.shape(gtids_map)[0]):
        pred_id = np.int(gtids_map[i,1])
        mask1 = np.isin(pred.flatten(), pred_id).reshape(pred.shape)
        mask_pred.append(mask1)
    
    if only_pred == False:
        for i in range(np.shape(gtids_map)[0]):
            gt_id = np.int(gtids_map[i,0])
            mask2 = np.isin(gt.flatten(), gt_id).reshape(gt.shape)
            mask_gt.append(mask2)
        
    return np.array(mask_gt).squeeze(), np.array(mask_pred).squeeze(), gtids_map

def convert2coco(seg_data, aff_pred):
    """ 
    Convert the grount truth segmentation and the corresponding predictions to a coco dataset
    to evaluate this dataset. The 3D volume is comparable to a video-type dataset and will therefore
    be converted as a video instance segmentation 
    input:
    output: coco_result_vid.json : This file will be written to your current directory and contains
                                    the metadata about the dataset. 
    """
    print('\t-Started')    
    (gt, pred) = seg_data
    coco_list = []
    num_frames = pred.shape[0]; im_h = pred.shape[1]; im_w = pred.shape[2]
    input_videoId = 0 # index of video
    
# create complete mapping of ids for gt and pred:
    print('\t-\tObtain ID map ..')
    id_map = obtain_id_map(gt, pred)
    num_instances = id_map.shape[0]

#     id_map = scrrtest ##########################TESTING PARAM
#     num_instances = 2 ######################################### TESTING PARAM: #############################
    
    ui1,uc1 = np.unique(gt,return_counts=True) #very expensive, not necessary
    print('\t-\tTotal number of instances:\tgt: {}\tpred: {}'.format(ui1.size, num_instances))
    
    print('\t-\tConvert instances to COCO format ..')
    for i in range(num_instances):
        print('\t-- Instance {} out of {}'.format(i+1, num_instances))
        
        print('\t-\tObtain mask of each ID ..')
        _, mask_pred, ids_map = obtain_masks(gt, pred, id_map[i])
        
        print('\t-\tObtain mean affinity ..')
        mean_aff_score = affinity_score(aff_pred, mask_pred) # confidence of this instance
        
        pred_catId = 1 if id_map[i,1] != 0 else 0 # category of instance

        pred_segm = np.array((mask_pred), dtype=np.uint8)#, order='f') # segmentation mask across frames, fortr. uint8 req.
        
        # Format conversion
        print('\t-\tConvert Format ..')
        res_dict = convert_format(input_videoId, mean_aff_score, pred_catId, pred_segm)
        coco_list.append(res_dict)
    
    print('\n\t-\tDump COCO object to json ..')
    filename = 'COCO_Lucchi_train_result.json'
    with open(filename, 'w') as json_file:
        json.dump(coco_list, json_file)
    print('\t-\tCOCO object to written to {}.'.format(filename))
    print('\t-Finished')


# Test the function above
# from T_eval_zudi import seg_iou3d, loadh5py, get_bb3d
# from evaluation_postprocessing import *
# from src import *

# p="/n/home00/nwendt/pytorch_connectomics/scripts/outputs/unetv3_mito_retrain/result_train/augmentation_4fold_mean/"
# f="2_0.060000_0.480000_150_0.200000_150_0.900000_0_0.500000_aff60_his256.h5" #old file without erosion
p="/n/home00/nwendt/IoUanalysis_tools/MitoDataanalysis/"
f="Lucchi_train_multi_augmean_zfilt1_erode5_binary_label_2_0.060000_0.480000_150_0.200000_150_0.900000_0_0.500000_aff60_his256.h5"
pred = loadh5py(p + f)

gt_path='/n/pfister_lab2/Lab/vcg_connectomics/mitochondria/Lucchi/label/train_label_ins_v2.h5'
gt = loadh5py(gt_path)

aff_path='/n/home00/nwendt/pytorch_connectomics/scripts/outputs/unetv3_mito_retrain/result_train/augmentation_4fold_mean/result.h5'
aff=loadh5py(aff_path, vol='vol0')

convert2coco((gt, pred), aff)


# # Create Validation file


def convert_to_validation(info, licence, videos, categories):
    res_dict = dict()
    res_dict['info'] = info
    res_dict['licences'] = licence
    res_dict['videos'] = videos
    res_dict['categories'] = categories
    
    return res_dict


# You can manually enter and complete the data here 
info = {}
info['description']="Lucchi Dataset train stack"
info['url']="n.a"
info['version']="n.a"
info['year']=9999
info['contributor']="n.a"
info['date_created']="n.a"

licences = []
licence = {}
licence['url']="n.a"
licence['id']=1
licence['name']="n.a"
licences.append(licence)

videos = []
video = {}
video['height']=768
video['width']=1024 
video['length']=165
video['date_captured']="n.a" 
video['flickr_url']=""
video['file_names']=[]
video['id']=0
video['coco_url']=""
videos.append(video)


categories = []
category = {}
category['supercategory']="cell"
category['id']=1
category['name']="mitochondria"
categories.append(category)


# Format conversion
print('\t-\tConvert Format ..')
valid_dict =convert_to_validation(info, licences, videos, categories)

coco_val_list = valid_dict

#     import pdb; pdb.set_trace()
print('\n\t-\tDump COCO object to json ..')
filename = 'COCO_Lucchi_train_valid.json'
with open(filename, 'w') as json_file:
    json.dump(coco_val_list, json_file)
print('\t-\tCOCO object to written to {}.'.format(filename))
print('\t-Finished')


# # Other
def load_inference_meta(T_analysis_file):
    gt_ids = []
    pred_scores = []
    with open(T_analysis_file, 'r') as file:
        for rows in file:
            els = rows.split(",")
            gt_ids.append(int(els[0]))
            pred_scores.append(float(els[2]))
    return gt_ids, pred_scores

if False: # outcomment
    file = 'gt_analysis_Lucchi_train_multi_augmean_zfilt1_erode5_binary_label_2_0.060000_0.480000_150_0.200000_150_0.900000_0_0.500000_aff60_his256'
    ids, pred = load_inference_meta(file)
    print(ids, pred)


# # Install cocoapi for video instance segmentation
# https://github.com/youtubevos/cocoapi.git


# # Evaluation script for video instance segmentation

gt_path = 'COCO_Lucchi_train_valid.json'
# Define evaluator
ytvosGt = YTVOS(gt_path)
# Load segmentation result in COCO format
det_path = 'COCO_Lucchi_train_result.json'
ytvosDt = ytvosGt.loadRes(det_path)

ytvosEval = YTVOSeval(ytvosGt, ytvosDt, 'segm') # 'bbox' or 'segm'
ytvosEval.params.vidIds = sorted(ytvosGt.getVidIds())
ytvosEval.evaluate()
ytvosEval.accumulate()
ytvosEval.summarize()

