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

from vol3d_eval import VOL3Deval
from vol3d_util import seg_iou3d_sorted, readh5_handle, readh5, writeh5, unique_chunk


##### 1. I/O
def get_args():
    parser = argparse.ArgumentParser(description = 'Evaluate the mean average precision score (mAP) of 3D segmentation volumes')
    parser.add_argument('-gt', '--gt-seg', type = str, default = '~/my_ndarray.h5',
                       help='path to ground truth segmentation result')
    parser.add_argument('-gtb', '--gt-bbox', type = str, default = '',
                       help='path to a txt containing the (seg id, bounding box, volume) for each gt')
    parser.add_argument('-p', '--predict-seg', type = str, default = '~/my_ndarray.h5',
                       help='path to predicted instance segmentation result')
    # either input the pre-compute prediction score
    parser.add_argument('-ps', '--predict-score', type = str, default = '',
                       help='path to a txt or h5 file containing the confidence score for each prediction')
    parser.add_argument('-pb', '--predict-bbox', type = str, default = '',
                       help='path to a txt containing the (seg id, bounding box, volume[optional]) for each prediction')

    parser.add_argument('-gp', '--group-pred', type = str, default = '',
                       help='file to divide prediction into groups')
    parser.add_argument('-gg', '--group-gt', type = str, default = '',
                       help='file to divide ground truth into groups')

    parser.add_argument('-th', '--threshold', type = str, default = '5e3, 3e4',
                       help='get threshold for volume range [possible to have more than 4 ranges, c.f. cocoapi]')
    parser.add_argument('-thc', '--threshold-crumb', type = int, default = 2000,
                       help='throw away the imcomplete small mito in the ground truth for a meaningful evaluation')

    parser.add_argument('-cz', '--chunk-size', type = int, default = 250,
                       help='for memory-efficient computation, how many slices to load')

    parser.add_argument('-o', '--output-name', type = str, default = '',
                       help='output name prefix')
    parser.add_argument('-dt', '--do-txt', type = int, default = 1,
                       help='output txt for iou results')
    parser.add_argument('-de', '--do-eval', type = int, default = 1,
                       help='do evaluation')
    parser.add_argument('-sl', '--slices', type = str, default = "-1",
                       help="slices to load, example: -sl '50, 350'")
    parser.add_argument('-db', '--debug', type = str, default = 'db.h5',
                       help='do debug')
    
    args = parser.parse_args()
    if args.output_name=='':
        args.output_name = args.predict_seg[:args.predict_seg.rfind('.')] 
    
    return args


def load_data(args, slices):
    # load data arguments
    pred_seg = readh5_handle(args.predict_seg)
    gt_seg = readh5_handle(args.gt_seg)
    if slices[1] == -1:
        slices[1] = gt_seg.shape[0]
    pred_bbox, gt_bbox = None, None
    if args.predict_bbox != '':
        pred_bbox = np.loadtxt(args.predict_bbox).astype(int)
    if args.gt_bbox != '':
        gt_bbox = np.loadtxt(args.gt_bbox).astype(int)

    # check shape match
    sz_gt = np.array(gt_seg.shape)
    sz_pred = pred_seg.shape
    if np.abs((sz_gt-sz_pred)).max()>0:
        raise ValueError('Warning: size mismatch. gt: {}, pred: '.format(sz_gt,sz_pred))

    if args.predict_score != '':
        print('\t\t Load prediction score')
        # Nx2: pred_id, pred_sc
        if '.h5' in args.predict_score:
            pred_score = readh5(args.predict_score)
        elif '.txt' in args.predict_score:
            pred_score = np.loadtxt(args.predict_score)
        else:
            raise ValueError('Unknown file format for the prediction score')

        if not np.any(np.array(pred_score.shape)==2):
            raise ValueError('The prediction score should be a Nx2 array')
        if pred_score.shape[1] != 2:
            pred_score = pred_score.T
    else: # default
        print('\t\t Assign prediction score')
        # assign same weight
        """
        ui = unique_chunk(pred_seg, slices, chunk_size = args.chunk_size, do_count = False)
        ui = ui[ui>0]
        pred_score = np.ones([len(ui),2],int)
        pred_score[:,0] = ui
        """
        # alternative: sort by size
        ui,uc = unique_chunk(pred_seg, slices, chunk_size = args.chunk_size)
        uc = uc[ui>0]
        ui = ui[ui>0]
        pred_score = np.ones([len(ui),2],int)
        pred_score[:,0] = ui
        pred_score[:,1] = uc
    
    th_group, areaRng = np.zeros(0), np.zeros(0)
    if args.group_gt != '': # exist gt group file
        group_gt = np.loadtxt(args.group_gt).astype(int)
        group_pred = np.loadtxt(args.group_pred).astype(int)
    else:
        thres = np.fromstring(args.threshold, sep = ",")
        areaRng = np.zeros((len(thres)+2,2),int)
        areaRng[0,1] = 1e10
        areaRng[-1,1] = 1e10
        areaRng[2:,0] = thres
        areaRng[1:-1,1] = thres

    return gt_seg, pred_seg, pred_score, group_gt, group_pred, areaRng, slices, gt_bbox, pred_bbox

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
    print('\t1. Load data')
    args = get_args()
    
    if args.debug != '' and os.path.exists(args.debug):
        result_p, result_fn, pred_score_sorted, group_gt, group_pred, areaRng = readh5(args.debug, ['p','fn','score','g_gt', 'g_pred','ang'])
    else:
        def _return_slices():
            # check if args.slices is well defined and return slices array [slice1, sliceN]
            if args.slices == "-1":
                slices = [0, -1]
            else: # load specific slices only
                try:
                    slices = np.fromstring(args.slices, sep = ",", dtype=int)
                     #test only 2 boundaries, boundary1<boundary2, and boundaries positive
                    if (slices.shape[0] != 2) or \
                        slices[0] > slices[1] or \
                        slices[0] < 0 or slices[1] < 0:
                        raise ValueError("\nspecify a valid slice range, ex: -sl '50, 350'\n")
                except:
                    print("\nplease specify a valid slice range, ex: -sl '50, 350'\n")
            return slices
        slices = _return_slices()
        
        gt_seg, pred_seg, pred_score, group_gt, group_pred, areaRng, slices, gt_bbox, pred_bbox = load_data(args, slices)
        
        ## 2. create complete mapping of ids for gt and pred:
        print('\t2. Compute IoU')
        result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(pred_seg, gt_seg, pred_score, slices, group_gt, areaRng, args.chunk_size, args.threshold_crumb, pred_bbox, gt_bbox)
        stop_time = int(round(time.time() * 1000))
        print('\t-RUNTIME:\t{} [sec]\n'.format((stop_time-start_time)/1000) )
        if args.debug != '':
            writeh5(args.debug, [result_p, result_fn, pred_score_sorted, group_gt, group_pred, areaRng],['p','fn','score','g_gt', 'g_pred', 'ang'])

        ## 3. Evaluation script for 3D instance segmentation
    v3dEval = VOL3Deval(result_p, result_fn, pred_score_sorted, output_name=args.output_name)
    if args.do_txt > 0:
        v3dEval.save_match_p()
        v3dEval.save_match_fn()
    if args.do_eval > 0:
        print('start evaluation')        
        #Evaluation
        v3dEval.set_group(group_gt, group_pred)
        v3dEval.params.areaRng = areaRng
        v3dEval.accumulate()
        v3dEval.summarize()
        
if __name__ == '__main__':
    # python demo.py -gt /n/boslfs02/LABS/lichtman_lab/donglai/EM30/release/EM30-H-mito-test-v2.h5 -p tmp/0_human_instance_seg_pred.h5 -gg tmp/human_gt_stats_group.txt -gp tmp/pred_group.txt
    # python demo.py -gt /n/boslfs02/LABS/lichtman_lab/donglai/EM30/release/EM30-H-mito-test-v2.h5 -p tmp/ABCS/0_human_instance_seg_pred.h5 -gg tmp/human_gt_stats_group.txt -gp tmp/ABCS/ABCS_pred_threshold_file.txt
    main()
