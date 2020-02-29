#!/usr/bin/env python
# coding: utf-8

"""
This script allows you to obtain gt instance and prediction instance matches for the 3D mAP model evaluation. At the end, you can evaluate the mean average precision of your model based on the IoU metric. To do the evaluation, set evaluate to True (default).
"""

import time
import os, sys
import argparse
import numpy as np
import h5py

from vol3deval import VOL3Deval, seg_iou3d_sorted


##### 1. I/O
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the mean average precision score (mAP) of 3D segmentation volumes')
    parser.add_argument('-gt','--gt-seg', type=str, default='~/my_ndarray.h5',
                       help='path to ground truth segmentation result')

    parser.add_argument('-p','--predict-seg', type=str, default='~/my_ndarray.h5',
                       help='path to predicted instance segmentation result')
    # either input the pre-compute prediction score
    parser.add_argument('-ps','--predict-score', type=str, default='',
                       help='path to confidence score for each prediction')
    # or avg input affinity/heatmap prediction
    parser.add_argument('-ph','--predict-heatmap', type=str, default='',
                       help='path to heatmap for all predictions')
    parser.add_argument('-phc','--predict-heatmap-channel', type=int, default=-1,
                       help='heatmap channel to use')
    parser.add_argument('-th','--threshold', type=str, default='0, 1e10, 0, 1e5, 1e5, 5e5, 5e5, 1e10',
                       help='get threshold for volume range [possible to have more than 4 ranges, c.f. cocoapi]')

    parser.add_argument('-o','--output-name', type=str, default='vol3d',
                       help='output name prefix')
    parser.add_argument('-dt','--do-txt', type=int, default=1,
                       help='output txt for iou results')
    parser.add_argument('-de','--do-eval', type=int, default=1,
                       help='do evaluation')
    args = parser.parse_args()
    
    if args.predict_heatmap=='' and args.predict_score=='':
        raise ValueError('at least one of "predict_heatmap" and "predict_score" should not be zero')
    return args

def load_h5(path, vol=''):
    # do the first key
    fid = h5py.File(path, 'r')
    if vol == '': 
        if sys.version[0]=='3':
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

def main():
    """ 
    Convert the grount truth segmentation and the corresponding predictions to a coco dataset
    to evaluate this dataset. The 3D volume is comparable to a video-type dataset and will therefore
    be converted as a video instance segmentation 
    input:
    output: coco_result_vid.json : This file will be written to your current directory and contains
                                    the metadata about the dataset. 
    """
    ## 1. Load data
    start_time = int(round(time.time() * 1000))
    print('\n\t1. Load data')
    args = get_args()
    gt_seg, pred_seg, pred_score = load_data(args)
    areaRng = np.fromstring(args.threshold, sep = ",").reshape(-1, 2)
    
    ## 2. create complete mapping of ids for gt and pred:
    print('\n\t2. Compute IoU')
    result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(pred_seg, gt_seg, pred_score, areaRng)
    
    stop_time = int(round(time.time() * 1000))
    print('\t-RUNTIME:\t{} [sec]\n'.format((stop_time-start_time)/1000) )

#     print(result_p)
#     print(type(result_p))
#     print(result_p.shape)
    if args.do_txt == 1:
        header = '\tprediction  |\t\t gt all \t\t|\t\t gt small \t\t|\t\tgt medium \t\t|\t gt large\n' + \
        'ID, \tSIZE \t|\tID, SIZE, \tIoU \t|\tID, SIZE, \tIoU \t|\tID, SIZE, \tIoU \t|\tID, SIZE, \tIoU \t\n' + '-'*108
#         rowformat = '%d\t\t%4d\t\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f'    
        rowformat = '%d\t\t%4d\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f'        

        np.savetxt(args.output_name+'_p.txt', result_p, fmt=rowformat, header=header)

        header = '\t\t\t prediction \t\t |\t\t gt \t\t\n' + \
        'ID, \tSIZE | ID, SIZE, \tIoU \n' + '-'*40
        rowformat = '%d\t\t%4d\t\t%d\t%4d\t%.4f'
        np.savetxt(args.output_name+'_fn.txt', result_fn, fmt=rowformat, header=header)
        
    ## 3. Evaluation script for 3D instance segmentation
    if args.do_eval == 1:
        print('start evaluation')        
        #Evaluation
        v3dEval = VOL3Deval(result_p, result_fn, pred_score_sorted)
        #v3dEval.params.areaRng = [[0, 1e10], [0, 1e5], [1e5, 5e5], [5e5, 1e10]]
        v3dEval.params.areaRng = areaRng
        v3dEval.accumulate()
        v3dEval.summarize()
        

if __name__ == '__main__':
    main()

