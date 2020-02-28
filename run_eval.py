#!/usr/bin/env python
# coding: utf-8

"""
# Install cocoapi for video instance segmentation
https://github.com/ygCoconut/cocoapi.git (reforked from youtubevos)
This script allows you to obtain .json files in coco format from the ground truth instance segmentation array and the resulting instance prediction. At the end, you can evaluate the mean average precision of your model based on the IoU metric. To do the evaluation, set evaluate to True, which should be the case by default. 
"""
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# from pycocotools.ytvos import YTVOS
# from pycocotools.ytvoseval import YTVOSeval
from pycocotools.v3d import V3D
from pycocotools.v3deval import V3Deval

import pycocotools.mask as mask
import json
import h5py

import time
import os, sys
import argparse


##### 1. I/O
def is_python3():
    return sys.version[0]=='3'
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the mean average precision score (mAP) of 3D segmentation volumes')
    parser.add_argument('-p','--predict-seg', type=str, default='~/my_ndarray.h5',
                       help='path to predicted instance segmentation result')
    parser.add_argument('-gt','--gt-seg', type=str, default='~/my_ndarray.h5',
                       help='path to ground truth segmentation result')
    # either input the pre-compute prediction score
    parser.add_argument('-ps','--predict-score', type=str, default='',
                       help='path to confidence score for each prediction')
    # or avg input affinity/heatmap prediction
    parser.add_argument('-ph','--predict-heatmap', type=str, default='',
                       help='path to heatmap for all predictions')
    parser.add_argument('-phc','--predict-heatmap-channel', type=int, default=-1,
                       help='heatmap channel to use')
    parser.add_argument('-o','--output-name', type=str, default='coco',
                       help='output name prefix')
    parser.add_argument('-idmap','--get_idmap', type=str, default='True',
                       help='get id map to visualize data')
    parser.add_argument('-eval','--do_eval', type=str, default='True',
                       help='do evaluation')
    args = parser.parse_args()
    
    if args.predict_heatmap=='' and args.predict_score=='':
        raise ValueError('at least one of "predict_heatmap" and "predict_score" should not be zero')
    return args


def load_h5(path, vol=''):
    # do the first key
    fid = h5py.File(path, 'r')
    if vol == '': 
        if is_python3():
            vol = list(fid)[0]
        else: # python 2
            vol = fid.keys()[0] 
    return np.array(fid[vol]).squeeze()


def load_data(args):
    # load data arguments
    pred_seg = load_h5(args.predict_seg)
    gt_seg = load_h5(args.gt_seg)
    if args.predict_score != '':
        # Nx2: pred_id, pred_sc
        pred_score = load_h5(args.predict_score)
    else:
        pred_score = load_h5(args.predict_heatmap)
        pred_score = heatmap_to_score(pred_seg, pred_score, args.predict_heatmap_channel)

    return gt_seg, pred_seg, pred_score


def writejson(coco_list, filename):        
    with open(filename, 'w') as json_file:
        json.dump(coco_list, json_file)
    print('\t-\tCOCO object written to {}.'.format(filename))



###### 2. Seg IoU ID map
def get_bb3d(seg,do_count=False, uid=None):
    """returns bounding box of segments for higher processing speed
    Used for seg_iou3d."""
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
    # returns the matching pairs of ground truth IDs and prediction IDs, as well as the IoU of each pair.
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


def heatmap_to_score(pred, heatmap, channel=-1):
    if heatmap.ndim>pred.ndim:
        if channel != -1:
            heatmap = heatmap[channel]
        else:
            heatmap = heatmap.mean(axis=0)
    pred_id = np.unique(pred)
    pred_view = pred.ravel()
    pred_len = pred_id.max()+1
    
    # relabel bincount(minlen = max_len) with ids
    counts = np.bincount(pred_view, minlength=pred_len)
    sums = np.bincount(pred_view, weights=heatmap.ravel(), minlength=pred_len)
    return np.vstack([pred_id,(sums[pred_id]/counts[pred_id])/255.0]).T 

    
def obtain_id_map(gt, pred):
    """create complete mapping of ids for gt and pred pairs:"""
    # 1. get matched pair of ids
    gtids_map, ui2, uc2 = seg_iou3d(gt, pred, return_extra=True)
    gtids_map = gtids_map[:,[0,1,4,2,3]]
    
    # 2. get false positives
    false_positives = ui2[np.isin(ui2, gtids_map[:,1], assume_unique=True, invert=True)]
    #use hstack and vstack for speedup
    fp_stack = np.zeros((len(false_positives),5),int)
    fp_stack[:,1] = false_positives #insert false positives
    fp_stack[:,4] = uc2[np.isin(ui2, false_positives)] #insert false positives, -1 for background
    full_map = np.vstack((gtids_map, fp_stack))

    return full_map


#### 3. COCO format 
def convert_format_pred(input_videoId, pred_score, pred_seg):
    pred_dict = dict()
    pred_dict['video_id'] = input_videoId
    
    pred_dict['score'] = float(pred_score)
    pred_dict['category_id'] = int(1) # we only have instacnes of the same category
    pred_dict['segmentations'] = [None]*pred_seg.shape[0] #put all slices = None
    z_nonzero = np.max(np.max(pred_seg,axis=1),axis=1)
    for zid in np.where(z_nonzero>0)[0]:
        pred_dict['segmentations'][zid] = mask.encode(np.asfortranarray(pred_seg[zid]))
        if is_python3():
            pred_dict['segmentations'][zid]['counts'] = pred_dict['segmentations'][zid]['counts'].decode('ascii')
        
    return pred_dict

# Create GT file
def convert_format_gt(gt, gt_id):
    gt_dict = {}
    # move z axis to last dim in order to encode over z; mask.encode needs fortran-order array    
    gt_dict['segmentations'] = [None]*gt.shape[0]
    areas = np.sum(np.sum(gt,axis=1),axis=1)
    for zid in np.where(areas>0)[0]:
        gt_dict['segmentations'][zid] = mask.encode(np.asfortranarray(gt[zid]))
        if is_python3():
            gt_dict['segmentations'][zid]['counts'] = gt_dict['segmentations'][zid]['counts'].decode('ascii')
            
    gt_dict['height'] = gt.shape[1],
    gt_dict['width'] = gt.shape[2],
    gt_dict['length'] = gt.shape[0],
    gt_dict['length'] = 1,
    gt_dict['category_id'] = 1 # we only have instances of the same category
    gt_dict['id'] = int(gt_id)
    gt_dict['video_id'] = 0
    gt_dict['areas'] = areas.tolist()
    gt_dict['iscrowd'] = 0

    return gt_dict

#  get GT file meta data
def get_meta(data_size):
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
    video['height'] = data_size[1]
    video['width'] = data_size[2] 
    video['length'] = data_size[0]
    video['date_captured']="n.a" 
    video['flickr_url']=""
    video['file_names']=[]
    video['id']=0
    video['coco_url']=""
    videos.append(video)

    categories = []
    category = {}
    category['supercategory']="cell" 
    category['id']=int(1)
    category['name']="mitochondria"
    categories.append(category)
    
    gt_dict = dict()
    gt_dict['info'] = info
    gt_dict['licences'] = licence
    gt_dict['videos'] = videos
    gt_dict['categories'] = categories
    gt_dict['annotations'] = []
    
    return gt_dict 

    
def main(gt_seg, pred_seg, pred_score, output_name='coco'):
    """ 
    Convert the grount truth segmentation and the corresponding predictions to a coco dataset
    to evaluate this dataset. The 3D volume is comparable to a video-type dataset and will therefore
    be converted as a video instance segmentation 
    input:
    output: coco_result_vid.json : This file will be written to your current directory and contains
                                    the metadata about the dataset. 
    """
    print('\t-Started')    

    ui,uc = np.unique(gt_seg,return_counts=True)
    uc=uc[ui>0];ui=ui[ui>0]
    ui2,uc2 = np.unique(pred_seg,return_counts=True)
    uc2=uc2[ui2>0];ui2=ui2[ui2>0]
    
    input_videoId = 0 # index of single video = 3D volume
    num_gt_instances=ui.shape[0]
    num_pred_instances=ui2.shape[0]
    coco_list_pred = [None]*num_pred_instances # JSON prediction file made with list
    coco_dict_gt = get_meta(pred_seg.shape) # JSON GT file made with dict
    coco_dict_gt['annotations'] = [None]*num_gt_instances
    print('\t-\tTotal number of instances:\tgt: {}\tpred: {}'.format(num_gt_instances, num_pred_instances))
    
    print('\t-\tConvert instances to COCO format ..')
    for i in range(np.max([num_gt_instances, num_pred_instances])):
        print('\t-- Instance {} out of {}'.format(i+1, np.max([num_gt_instances, num_pred_instances])))
        
        ### gt instances
        if i < num_gt_instances:
            gt_id = ui[i]
            # coco format for gt
            coco_dict_gt['annotations'][i] = convert_format_gt((gt_seg==gt_id).astype(np.uint8), gt_id)

        #### predictions    
        if i < num_pred_instances:
            pred_id = ui2[i]
            pred_dict = convert_format_pred(input_videoId, pred_score[pred_score[:,0]==pred_id,1], (pred_seg==pred_id).astype(np.uint8))
            coco_list_pred[i] = pred_dict #pre-allocation is faster !

 
    coco_list_pred = [i for i in coco_list_pred if i] #remove empty None elements (if exist)
    coco_dict_gt['annotations'] = [i for i in coco_dict_gt['annotations'] if i]

    print('\n\t-\tWrite COCO object to json ..')
    writejson(coco_list_pred, filename = output_name+'_pred.json')
    writejson(coco_dict_gt, filename = output_name+'_gt.json')
    print('\t-Finished\n\n')
    return output_name+'_pred.json', output_name+'_gt.json'


if __name__ == '__main__':
    ## Create predict.json and gt.json by using functions above
    print('\nload data')
    args = get_args()
    gt_seg, pred_seg, pred_score = load_data(args)

    
    # create complete mapping of ids for gt and pred, then write it to json:
    if args.get_idmap == 'True':
        print('\n[optional]\tObtain ID map and bounding box ..')
        id_map = obtain_id_map(gt_seg, pred_seg)
        header ='load: np.loadtxt(\'id_map_iou.txt\')\n\n' + \
            'GT_id, pred_id, IoU, \t\tGT_sz, \t\tpred_sz\n' + \
            '-------------------------------------------'
        np.savetxt('id_map_iou.txt', id_map, fmt='%d\t\t%d\t\t%1.4f\t\t%4d\t\t%d', header=header)
        
    print('\ncreate coco file')
    start_time = int(round(time.time() * 1000))
    pred_json, gt_json = main(gt_seg, pred_seg, pred_score, args.output_name)
    stop_time = int(round(time.time() * 1000))
    print('\t-RUNTIME:\t{} [sec]\n'.format((stop_time-start_time)/1000) )

    # # Evaluation script for video instance segmentation
    if args.do_eval == 'True':
        print('start evaluation')
        gt_path = gt_json
        # Define evaluator
        v3dGt = V3D(gt_path)
        # Load segmentation result in COCO format
        det_path = pred_json
        v3dDt = v3dGt.loadRes(det_path)
        v3dEval = V3Deval(v3dGt, v3dDt, 'segm') # 'bbox' or 'segm'
        #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        v3dEval.params.areaRng = [[0, 1e10], [0, 1e5], [1e5, 5e5], [5e5, 1e10]] # [All, Small, Medium, Large]
        v3dEval.params.vidIds = sorted(v3dGt.getVidIds())
        v3dEval.evaluate()
        v3dEval.accumulate()
        v3dEval.summarize()
        
