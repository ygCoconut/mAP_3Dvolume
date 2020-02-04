#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval

import pycocotools.mask as mask
import json
import h5py

import os, sys
import argparse

# Arguments

parser = argparse.ArgumentParser(description='Evaluate the mean average precision score (mAP) of 3D segmentation volumes')

parser.add_argument('-p','--prediction_path', type=str, default='~/my_ndarray.h5')
parser.add_argument('-gt','--ground_truth_path', type=str, default='~/my_ndarray.h5')
parser.add_argument('-aff','--affinity_path', type=str, default='~/my_ndarray.h5')

parser.add_argument('-eval','--evaluate_map_score', type=bool, default=True)

args = parser.parse_args()

# JSON file creation arguments
prediction = args.prediction_path
ground_truth= args.ground_truth_path
affinity = args.affinity_path

evaluate = args.evaluate_map_score


# # How to convert video instance segmentation results into COCO format

def get_bb3d(seg,do_count=False, uid=None):
    """returns bounding box of segments for higher processing speed
    Used for seg_iou3d. Written by Donglai Wei"""
    sz = seg.shape
    assert len(sz)==3
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid>0]
    um = int(uid.max())
    out = np.zeros((1+um,7+do_count),dtype=np.uint32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1] = sz[0]
    out[:,3] = sz[1]
    out[:,5] = sz[2]

    # for each slice
    zids = np.where((seg>0).sum(axis=1).sum(axis=1)>0)[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,1] = np.minimum(out[sid,1],zid)
        out[sid,2] = np.maximum(out[sid,2],zid)

    # for each row
    rids = np.where((seg>0).sum(axis=0).sum(axis=1)>0)[0]
    for rid in rids:
        sid = np.unique(seg[:,rid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,3] = np.minimum(out[sid,3],rid)
        out[sid,4] = np.maximum(out[sid,4],rid)
    
    # for each col
    cids = np.where((seg>0).sum(axis=0).sum(axis=0)>0)[0]
    for cid in cids:
        sid = np.unique(seg[:,:,cid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,5] = np.minimum(out[sid,5],cid)
        out[sid,6] = np.maximum(out[sid,6],cid)

    if do_count:
        ui,uc = np.unique(seg,return_counts=True)
        out[ui[ui<=um],-1]=uc[ui<=um]

    return out[uid]

def seg_iou3d(seg1, seg2, return_extra=False):
    """returns the matching pairs of ground truth IDs and prediction IDs, as well as the IoU of each pair.
    Written by Donglai Wei"""
    # (gt,pred)
    # return: id_1,id_2,size_1,size_2,iou
    ui,uc = np.unique(seg1,return_counts=True)
    uc=uc[ui>0];ui=ui[ui>0]
    ui2,uc2 = np.unique(seg2,return_counts=True)
    uc2=uc2[ui2>0];ui2=ui2[ui2>0]

    out = np.zeros((len(ui),5),float)
    bbs = get_bb3d(seg1,uid=ui)[:,1:]
    out[:,0] = ui
    out[:,2] = uc

    for j,i in enumerate(ui):
        bb= bbs[j]
        ui3,uc3=np.unique(seg2[bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(seg1[bb[0]:bb[1]+1,bb[2]:bb[3]+1]==i),return_counts=True)
        uc3[ui3==0]=0
        # take the largest one
        out[j,1] = ui3[np.argmax(uc3)] # matched seg id (max)
        if out[j,1]>0:
            out[j,3] = uc2[ui2==out[j,1]] # matched seg size
            out[j,4] = float(uc3.max())/(out[j,2]+out[j,3]-uc3.max()) # iou
    if return_extra: # for FP
        return out,ui2,uc2
    else:
        return out


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


def convert_format_pred(input_videoId, pred_score, pred_catId, pred_segm):
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
    

def convert2coco(seg_data, aff_pred, convert_gt=True):
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
    gt_list = []
    num_frames = pred.shape[0]; im_h = pred.shape[1]; im_w = pred.shape[2]
    input_videoId = 0 # index of video
    
# create complete mapping of ids for gt and pred:
    print('\t-\tObtain ID map ..')
    id_map = obtain_id_map(gt, pred)
    num_instances = id_map.shape[0]

#     id_map = scrrtest ##########################TESTING PARAM
    num_instances = 4 ######################################### TESTING PARAM: #############################
    
    ui1,uc1 = np.unique(gt,return_counts=True) #very expensive, not necessary
    print('\t-\tTotal number of instances:\tgt: {}\tpred: {}'.format(ui1.size, num_instances))
    
    print('\t-\tConvert instances to COCO format ..')

    gt_dict = get_meta()
    for i in range(num_instances):
        print('\t-- Instance {} out of {}'.format(i+1, num_instances))

        print('\t-\tObtain mask of each ID ..')
        mask_gt, mask_pred, ids_map = obtain_masks(gt, pred, id_map[i], only_pred=False)

        print('\t-\tObtain mean affinity ..')
        mean_aff_score = affinity_score(aff_pred, mask_pred) # confidence of this instance

        pred_catId = 1 if id_map[i,1] != 0 else 0 # category of instance

        pred_segm = np.array((mask_pred), dtype=np.uint8)#, order='f') # segmentation mask across frames, fortr. uint8 req.
        
        # Format conversion
        print('\t-\tConvert Format ..')
        res_dict = convert_format_pred(input_videoId, mean_aff_score, pred_catId, pred_segm)
        coco_list.append(res_dict)
        
        if convert_gt == True:
#             gt_dict = convert_format_gt(gt_dict, np.array((mask_gt), dtype=np.uint8))
            gt_dict = convert_format_pred(input_videoId, 1.0, pred_catId, np.array((mask_gt), dtype=np.uint8))
            gt_list.append(gt_dict)
        

    print('\n\t-\tDump COCO object to json ..')
    writejson(coco_list, filename = 'COCO_segmentation_traindata_result.json')
    writejson(gt_list, filename = 'COCO_segmentation_traindata_gt.json')
    print('\t-Finished\n\n')

    
def writejson(coco_list, filename):        
    with open(filename, 'w') as json_file:
        json.dump(coco_list, json_file)
    print('\t-\tCOCO object to written to {}.'.format(filename))


# # Create Validation file
def convert_format_gt(res_dict, gt):
    annotation_dict = {}
    # move z axis to last dim in order to encode over z; mask.encode needs fortran-order array    
    encoded = mask.encode(np.asfortranarray(np.moveaxis(gt, 0, -1)))
    for i in range(len(encoded)):
        # python2 and python3 bug with bytes and strings to be avoided for "counts"
        encoded[i]['counts'] = encoded[i]['counts'].decode('ascii')
        # make z-slices without the specific instance None
        if np.sum(gt[i]) == 0:
            encoded[i] = None
            
    annotation_dict['segmentations'] = encoded
    res_dict['annotations']['height'] = 720,
    res_dict['annotations']['width'] = 1280,
    res_dict['annotations']['length'] = 1,
    res_dict['annotations']['category_id'] = 1
    res_dict['annotations']['video_id'] = 0
    
    return res_dict['annotations'].append(annotation_dict)
#     res_dict['annotations']['segmentations'] = encoded
#     res_dict['annotations']['height'] = 720,
#     res_dict['annotations']['width'] = 1280,
#     res_dict['annotations']['length'] = 1,
#     res_dict['annotations']['category_id'] = 1
#     res_dict['annotations']['video_id'] = 0
    
#     return res_dict

def get_meta():
    # You can manually enter and complete the data here
    root_dict = dict()
                                    
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
    
    res_dict = dict()
    res_dict['info'] = info
    res_dict['licences'] = licence
    res_dict['videos'] = videos
    res_dict['categories'] = [categories]
    
    res_dict['annotations'] = []
#     res_dict['annotations']['segmentations'] = [[]]
    
    return res_dict 

## Create predict.json and gt.json by using functions above

pred = loadh5py(prediction)
gt = loadh5py(ground_truth)
aff=loadh5py(affinity, vol='vol0')
convert2coco((gt, pred), aff)

def debugjson():
    annFile_foot = 'COCO_segmentation_traindata_gt.json' 
    annFile_foot_modified = 'COCO_segmentation_traindata_gt_modified.json'

    with open(annFile_foot) as f:
        data = json.loads(f.read())

        #add additional brackets to categories
        data['categories']=[data['categories']] 

        #add additional brakets to annotations
        for i in range(len(data['annotations'])):
            if type(data['annotations'][i]['segmentation'][0])!=list:
                data['annotations'][i]['segmentation'] = [data['annotations'][i]['segmentation']] #additional brackets

        #export
        with open(annFile_foot_modified, 'w+') as ff:
            ff.write(json.dumps(data))

# Execute function above:
# validation_data_to_json()

# https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1044
# debugjson()


# # Install cocoapi for video instance segmentation
# https://github.com/youtubevos/cocoapi.git

if evaluate == True:
    # # Evaluation script for video instance segmentation

    gt_path = 'COCO_segmentation_traindata_gt.json'
    # Define evaluator
    ytvosGt = YTVOS(gt_path)
    # Load segmentation result in COCO format
    det_path = 'COCO_segmentation_traindata_result.json'
    ytvosDt = ytvosGt.loadRes(det_path)

    ytvosEval = YTVOSeval(ytvosGt, ytvosDt, 'segm') # 'bbox' or 'segm'
    ytvosEval.params.vidIds = sorted(ytvosGt.getVidIds())
    ytvosEval.evaluate()
    ytvosEval.accumulate()
    ytvosEval.summarize()

