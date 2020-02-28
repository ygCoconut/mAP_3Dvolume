#!/usr/bin/env python
# coding: utf-8

"""
# Install cocoapi for video instance segmentation
https://github.com/youtubevos/cocoapi.git
This script allows you to obtain .json files in coco format from the ground truth instance segmentation array and the resulting instance prediction. At the end, you can evaluate the mean average precision of your model based on the IoU metric. To do the evaluation, set evaluate to True, which should be the case by default. 
"""
import numpy as np

from cocoevalShort import YTVOSeval

import json
import h5py

import time
import os, sys
import argparse


##### 1. I/O

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
    parser.add_argument('-th','--threshold', type=str, default='0, 1e10, 0, 1e5, 1e5, 5e5, 5e5, 1e10',
                       help='get threshold for volume range [possible to have more than 4 ranges, c.f. cocoapi]')
    parser.add_argument('-json','--writejson', type=str, default='False',
                       help='Boolean to write a coco json file')
    args = parser.parse_args()
    
    if args.predict_heatmap=='' and args.predict_score=='':
        raise ValueError('at least one of "predict_heatmap" and "predict_score" should not be zero')
    return args

def is_python3():
    return sys.version[0]=='3'

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
    count_voxel = np.bincount(pred_view, minlength=pred_len)
    count_score = np.bincount(pred_view, weights=heatmap.ravel(), minlength=pred_len)
    score = count_score[pred_id]/count_voxel[pred_id]
    if score.max()>1.1:#assume 0-255
        score = score/255.
    out = np.vstack([pred_id,score]).T 
    return out

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

def seg_iou3d(seg1, seg2, th, uid=None, return_extra=False):
    # returns the matching pairs of ground truth IDs and prediction IDs, as well as the IoU of each pair.
    # (pred,gt)
    # return: id_1,id_2,size_1,size_2,iou
    ui,uc = np.unique(seg1,return_counts=True)
    uc=uc[ui>0];ui=ui[ui>0]
    
    ui2,uc2 = np.unique(seg2,return_counts=True)
    uc2=uc2[ui2>0];ui2=ui2[ui2>0]
    
    if uid is None:
        uid = ui
        uic = uc
    else:
        uc_rl = np.zeros(ui.max()+1,int)
        uc_rl[ui] = uc
        uic = uc_rl[uid]            
    
    bbs = get_bb3d(seg1,uid=uid)[:,1:]    
    
    p_stack = np.zeros((len(uid), 2+3*th.shape[0]), float)
    p_stack[:,0] = uid
    p_stack[:,1] = uic

    
    seg2_id = np.zeros(1+ui2.max(), int)
    seg2_iou = np.zeros(1+ui2.max(), float)
    for j,i in enumerate(uid):
        # Find intersection of pred and gt instance inside bbox, call intersection ui3
        bb= bbs[j]
        ui3,uc3=np.unique(seg2[bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(seg1[bb[0]:bb[1]+1,bb[2]:bb[3]+1]==i),return_counts=True)
        uc3=uc3[ui3>0] # get intersection counts
        ui3=ui3[ui3>0] # get intersection ids        
        if len(ui3)>0:
            # get count of all preds inside bbox (assume ui2,ui3 are of ascending order)
            uc2_match = uc2[np.isin(ui2,ui3)]
            #IoUs = A + B - C = uc[j] + uc2_subset - uc3
            ious = uc3.astype(float)/(uic[j] + uc2_match - uc3) #all possible iou combinations of bbox ids are contained
            
            for r in range(th.shape[0]): # fill up all, then s, m, l
                gid = (uc2_match>th[r,0])*(uc2_match<=th[r,1])
                if sum(gid)>0: 
                    idx_iou_max = np.argmax(ious*gid)
                    p_stack[j,2+r*3:2+r*3+3] = [ ui3[idx_iou_max], uc2_match[idx_iou_max], ious[idx_iou_max] ]            
            # update set2
            seg2_todo = seg2_iou[ui3]<ious            
            seg2_iou[ui3[seg2_todo]] = ious[seg2_todo]
            seg2_id[ui3[seg2_todo]] = i
                
    # get the rest: false negative + dup
    fn_gid = ui2[np.isin(ui2, p_stack[:,2], assume_unique=False, invert=True)]
    fn_gic = uc2[np.isin(ui2, fn_gid)]
    fn_iou = seg2_iou[fn_gid]
    fn_pid = seg2_id[fn_gid]
    fn_pic = np.hstack([uc[np.isin(ui, fn_pid)],np.zeros((fn_pid==0).sum())])
    
    # add back duplicate
    # instead of bookkeeping in the previous step, faster to redo them    
    fn_stack = np.vstack([fn_pid, fn_pic, fn_gid, fn_gic,fn_iou]).T
    
    return p_stack, fn_stack

   
def obtain_id_map(gt, pred, scores, th):
    """create complete mapping of ids for gt and pred pairs:"""
    
    # 1. get matched pair of ids    
    rl = np.zeros(int(np.max(scores[:,0])+1), float)
    rl[scores[:,0].astype(int)] = scores[:,1]
    
    # 1. sort the prediction by confidence
    ui = np.unique(pred);ui=ui[ui>0]    
    sid = np.argsort(-rl[ui])
    
    p_map, fn_map = seg_iou3d(pred, gt, th, uid=ui[sid])
    # format: pid,pc,p_score, gid,gc,iou
    scores_out = rl[ui[sid]].reshape(-1,1)
    return np.hstack([p_map[:,:2], scores_out, p_map[:,2:]]), fn_map
                     
def main():
    """ 
    Convert the grount truth segmentation and the corresponding predictions to a coco dataset
    to evaluate this dataset. The 3D volume is comparable to a video-type dataset and will therefore
    be converted as a video instance segmentation 
    input:
    output: coco_result_vid.json : This file will be written to your current directory and contains
                                    the metadata about the dataset. 
    """
    ## 1. Initialization
    print('\t-Started')    
    start_time = int(round(time.time() * 1000))
    print('\n\t-Load data')
    args = get_args()
    th = np.fromstring(args.threshold, sep = ",").reshape(-1, 2)
    gt_seg, pred_seg, pred_score = load_data(args)
    
    ## 2. create complete mapping of ids for gt and pred:
    print('\n\t-Obtain ID map and bounding box ..')
    p_map, fn_map = obtain_id_map(gt_seg, pred_seg, pred_score, th)
    
    stop_time = int(round(time.time() * 1000))
    print('\t-RUNTIME:\t{} [sec]\n'.format((stop_time-start_time)/1000) )
    if args.get_idmap == 'True':
        header ='load: np.loadtxt(\'p_map_iou.txt\')\n\n' + \
        '\t\t\t prediction \t\t\t\t |\t\t gt all \t\t|\t gt small \t\t|\t gt medium \t\t|\t gt large\n' + \
        'ID, \tSIZE, SCORE  | ID, SIZE, \tIoU | ID, SIZE, \tIoU | ID, SIZE, \tIoU | ID, SIZE, \tIoU\n' + '-'*116
        rowformat = '%d\t\t%4d\t\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f'        
        np.savetxt('iou_p.txt', p_map, fmt=rowformat, header=header)

        header ='load: np.loadtxt(\'p_map_iou.txt\')\n\n' + \
        '\t\t\t prediction \t\t |\t\t gt \t\t\n' + \
        'ID, \tSIZE | ID, SIZE, \tIoU \n' + '-'*40
        rowformat = '%d\t\t%4d\t\t%d\t%4d\t%.4f'
        np.savetxt('iou_fn.txt', fn_map, fmt=rowformat, header=header)
        
    ## 3. Evaluation script for 3D instance segmentation
    if args.do_eval == 'True' and args.get_idmap == 'True':
        print('start evaluation')        
        #Evaluation
        ytvosEval = YTVOSeval(p_map, fn_map, 'segm') # 'bbox' or 'segm'
        # Default thresholds: [All, Small, Medium, Large] = [[0, 1e10], [0, 1e5], [1e5, 5e5], [5e5, 1e10]]        
        #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        #ytvosEval.params.areaRng = [[0, 1e10], [0, 1e5], [1e5, 5e5], [5e5, 1e10]]
        th = np.fromstring(args.threshold, sep = ",").reshape(-1, 2)
        ytvosEval.params.areaRng = th
        
        #These two will be written into this file here.
        ytvosEval.accumulate()
        ytvosEval.summarize()
        
    else: print("make sure the flags do_eval and get_idmap are set !")

        

if __name__ == '__main__':
    main()


################################################# DEBUG
    if False:
        
        from pycocotools.coconut import YTVOS
        from pycocotools.coconuteval import YTVOSeval
        import numpy as np
        import pickle
        def save_dict(filename_, di_):
            with open(filename_, 'wb') as f:
                pickle.dump(di_, f)

        def load_dict(filename_):
            with open(filename_, 'rb') as f:
                ret_di = pickle.load(f)
            return ret_di
        
#         save_dict('gt.npy', coco_dict_gt)
        
        
        coco_dict_gt = load_dict('gt.npy')
        coco_list_pred= np.load('pred.npy', allow_pickle = True).tolist()
        
        
        print('start evaluation')
        ytvosGt = coco_dict_gt
        ytvosDt = coco_list_pred

        ytvosEval = YTVOSeval(ytvosGt, ytvosDt, 'segm') # 'bbox' or 'segm'
        #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        ytvosEval.params.areaRng = [[0, 1e10], [0, 1e5], [1e5, 5e5], [5e5, 1e10]] # [All, Small, Medium, Large]
        
        ytvosEval.accumulate()
        ytvosEval.summarize()