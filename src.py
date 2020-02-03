


# T_eval_zudi.py ############################################

import os, sys
import numpy as np
import h5py
# from T_util import writeh5
import argparse

def get_bb3d(seg,do_count=False, uid=None):
#      This function has been written by Donglai Wei.
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

def seg_iou3d(seg1, seg2):
#      This function has been written by Donglai Wei.
    # (gt,pred)
    # return: id_1,id_2,size_1,size_2,iou
    ui,uc = np.unique(seg1,return_counts=True)
    uc=uc[ui>0];ui=ui[ui>0]
    ui2,uc2 = np.unique(seg2,return_counts=True)
    uc=uc[ui>0];ui=ui[ui>0]

    out = np.zeros((len(ui),5),float)
    bbs = get_bb3d(seg1,uid=ui)[:,1:]
#     import pdb; pdb.set_trace()
    out[:,0] = ui
    out[:,2] = uc
    for j,i in enumerate(ui):
        bb= bbs[j]
        ui3,uc3=np.unique(seg2[bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(seg1[bb[0]:bb[1]+1,bb[2]:bb[3]+1]==i),return_counts=True)
        uc3[ui3==0]=0
        # take the largest one
        out[j,1] = ui3[np.argmax(uc3)] # matched seg id (max)
        out[j,3] = uc2[ui2==out[j,1]] # matched seg size
        out[j,4] = float(uc3.max())/(out[j,2]+out[j,3]-uc3.max()) # iou
    
    return out


# IoU_rank-Nils.py @###############################################################

# from T_eval_zudi import seg_iou3d, loadh5py, get_bb3d
# from evaluation_postprocessing import *

def IoU_eval(pred_path,gt_path):
    """
    Short description: Same as IoU_rank.py, but with IoU as final output.
    Long one: After computing the IoU score for multiple instances, we are interested in evaluating the binary IoU score, given that we perfectly segment a specific instance.
    1) For every ID: 
        a) replace pred with gt
        b) calc binary IoU
        c) calc previous binary IoU - new binary IoU ( with error correction )
    2) Order segments by biggest error difference.
    3) You can now analyse the erroneous IDs and their error sources to determine by how much the IoU is going to improve, given that certain instances would be perfectly segmented.
    """

    pred=loadh5py(pred_path)
    gt = loadh5py(gt_path)

    gt_bin = multi2binlabel(gt)
    bin_IoU = seg_iou3d(gt_bin, multi2binlabel(pred))
    old_IoU = np.float(bin_IoU[:,4])
    instance_IoU = seg_iou3d(gt, pred)

    # old_IoU[:0] = gt_id
    # old_IoU[:1] = corresponding pred_id
    # 1) For i in gt_id: 
    #     make pred_id == 0
    # 2) For i in gt_id:
    #     insert gt labelings into pred
    # calc new IoU

    diff_list = []
    for i in range(np.shape(instance_IoU)[0]):
        if i == 8:
            import pdb; pdb.set_trace()
        pred_tmp = pred.copy()

        gt_id = np.int(instance_IoU[i,0])
        pred_id = np.int(instance_IoU[i,1])
        bb = bbs[i]

        gt_new = gt[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]]
        pred_new = pred[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]]

    #     supposes your prediction ID mask mask1 only contains 1 single GT foreground ID
        mask1 = np.isin(pred_new.flatten(), pred_id).reshape(pred_new.shape)
        pred_new[mask1] = 0
        mask2 = np.isin(gt_new.flatten(), gt_id).reshape(gt_new.shape)
        pred_new[mask2] = pred_id
        pred_tmp[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = pred_new

        diff_list.append([gt_id, pred_id, 
                          np.float(seg_iou3d(gt_bin, 
                                    multi2binlabel(pred_tmp))[:,4]) - old_IoU])

    return sorted(diff_list, key=lambda i: i[2], reverse=True)



def _get_gtmap(gt, pred):
    return seg_iou3d(gt, pred)[:,:2]

def cumulative_iou():
    delta = "12.0 0.00508108,10.0 0.0050263,16.0 0.00493998,19.0 0.00454979,35.0 0.00411105,15.0 0.00340789,18.0 0.00313245,23.0 0.00298811,38.0 0.00293631,6.0 0.00279911,26.0 0.00221369,2.0 0.00215515,22.0 0.00213399,31.0 0.00183232,27.0 0.00164086,34.0 0.00162723,24.0 0.00156285,9.0 0.00153079,3.0 0.00143954,40.0 0.00113878,13.0 0.00111814,8.0 0.00099495,43.0 0.00096601,36.0 0.00087411,14.0 0.00086255,30.0 0.00079466,11.0 0.00062981,4.0 0.00062228,21.0 0.00061631,32.0 0.00058036,20.0 0.00054232,41.0 0.00049027,25.0 0.00047184,1.0 0.00044056,5.0 0.00027934,17.0 0.00015096,29.0 0.00011864,7.0 9.70939823e-05"
    a = delta.split(",")
    b = [i.split(' ') for i in a]
    c = [float(i[1]) for i in b]
    d=c
    for i in range(1, len(list(d))):
        d[i] += d[i-1]
    print(d)

    
def insertGT(gt, pred, gt_ids_list=None):
    """This function takes label prediction and label ground truth of a 3D vol as input and returns """
    gtids_map = _get_gtmap(gt, pred)
    
    if gt_ids_list is not None:
        mask0 = np.isin(gtids_map[:,0].flatten(), gt_ids_list).reshape(gtids_map[:,0].shape)
        gtids_map = gtids_map[mask0]
    print("the following ID map will be used:\n [ gt --> pred ]\n ---------------\n", gtids_map)
    for i in range(np.shape(gtids_map)[0]):
        gt_id = np.int(gtids_map[i,0])
        pred_id = np.int(gtids_map[i,1])
        mask1 = np.isin(pred.flatten(), pred_id).reshape(pred.shape)
        pred[mask1] = 0
        
    for i in range(np.shape(gtids_map)[0]):
        gt_id = np.int(gtids_map[i,0])
        pred_id = np.int(gtids_map[i,1])       
        mask2 = np.isin(gt.flatten(), gt_id).reshape(gt.shape)
        pred[mask2] = pred_id
        
    return pred




# Evaluation_postprocessing.py ########################################################

import numpy as np
import argparse
import h5py
import time
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, erosion, cube, remove_small_objects

def loadh5py(path, vol="main"):
    return np.array(h5py.File(path, 'r')[vol]).squeeze()

def writeh5_file(file, filename=None):
    if filename is None:
        filename="{}.h5".format(file)
    hf = h5py.File(filename, 'w')
    hf.create_dataset('main', data=file)
    hf.close()

def multi2binlabel(h5_or_np_data):
    data = np.array(h5_or_np_data).astype(np.uint16).squeeze()
    data[data > 0] = 1
    return data

def erodevols(h5_or_np_data, reps = 1):
    """In-plane erosion with a cross-shaped SE"""
    data = np.array(h5_or_np_data).astype(np.uint16).squeeze()
    if reps > 1:
        for i in range(reps - 1):
            data = erosion(data)
    return erosion(data)

def erodevols_bin(h5_or_np_data, reps = 1):
    """Faster erosion for b&w data. Potential disadvantage:
    Non-Erosion of falsely collated regions of different segments"""
    data = np.array(h5_or_np_data).astype(np.uint16).squeeze()
    if reps > 1:
        for i in range(reps - 1):
            data = binary_erosion(data)
    return binary_erosion(data)

def z_filter(h5_or_np_data, num_slices=1):
    """Deletes the segments present in less than num_slices 
    consecutive planes along the z-axis.
    Eliminates "crumbs" with low depth.
    Idea from "Fast Mitochondria Segmentation for Connectomics",
    Casser et al."""
    data = np.array(h5_or_np_data).astype(np.uint16).squeeze()
    
    label_collection = np.unique(data)
    label_counter = np.zeros(label_collection.size)#, dtype=uint)
    sz = data.shape[0]
#  Count number of slices containing a certain ID
    for i in range(sz):
        curr_labels = np.unique(data[i])
        for k in curr_labels:
            label_counter[label_collection == k] += 1
# Filter crumbs out    
    crumb_IDs = label_collection[label_counter <= num_slices]
    mask = np.isin(data.flatten(), crumb_IDs).reshape(data.shape)
    data[mask] = 0
    return data


# File needed to postprocess the model output.
def postprocessing(path, file, num_slices=0, erodereps=0, dstype='train'):
    data = loadh5py(path + file)
    data = z_filter(data, num_slices=num_slices)
    data = erodevols(data, reps=erodereps)
#     data = multi2binlabel(data)
    writeh5_file(data,
         "Lucchi_{}_augmean_zfilt{}_erode{}_binary_label_{}".format(dstype,
                                                                num_slices, 
                                                                erodereps, file))

# visualisation of the data distribution
def sort_segments_by_volume(volume_path):
    """ returns the sorted list of ids, from smallest volumes to the biggest"""
    data = np.array(h5py.File(volume_path, 'r')['main']).flatten()
    ids, num_voxels = np.unique(data, return_counts=True)
    sid = np.argsort(num_voxels)
    return ids[sid], num_voxels[sid]

def visualize_volumes(volume, a=0, b=None, save=False):
    """returns a plot of the size-sorted ids of a segmented volume
    Options: give range [a b] of IDs to inspect, save figure"""
    ids, segment_volumes = sort_segments_by_volume(volume)
    if b==None:
        b=np.size(segment_volumes)
    plt.plot(segment_volumes[a:b])
    plt.title('Number of voxels per segment ID')
    plt.xlabel('segments')
    plt.ylabel('# of voxels')
    if save:
        plt.savefig('voxelsize_range{}:{}.svg'.format(a,b-1))