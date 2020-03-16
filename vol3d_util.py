import sys
import numpy as np
import h5py
import cv2
from tqdm import tqdm

####
# list of utility functions
# 0. I/O util
# 1. binary pred -> instance seg
# 2. instance seg + pred heatmap -> instance score
# 3. instance seg -> bbox
# 4. instance seg + gt seg + instance score -> sorted match result

# 0. I/O
def seg2im(seg): # seg -> 3-channel image
    if seg.max()>255:
        return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)
    else:
        return seg.astype(np.uint8)

def im2seg(im): # image -> seg
    if im.ndim==2:
        return im
    else:
        return im[:,:,0].astype(np.uint32)*65536+im[:,:,1].astype(np.uint32)*256+im[:,:,2].astype(np.uint32)

def heatmap_by_channel(im, channel=-1): # image to heatmap
    if channel != -1:
        heatmap = im[channel]
    else:
        heatmap = im.mean(axis=0)
    return heatmap

def readh5(path, vol=''):
    # do the first key
    fid = h5py.File(path, 'r')
    if vol == '': 
        if sys.version[0]=='3':
            vol = list(fid)[0]
        else: # python 2
            vol = fid.keys()[0] 
    return np.array(fid[vol]).squeeze()

# 1. binary pred -> instance seg
def seg_bbox2d(seg,do_count=False, uid=None):
    sz = seg.shape
    assert len(sz)==2
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid>0]

    um = uid.max()
    out = np.zeros((1+int(um),5+do_count),dtype=np.uint32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1] = sz[0]
    out[:,3] = sz[1]
    # for each row
    rids = np.where((seg>0).sum(axis=1)>0)[0]
    for rid in rids:
        sid = np.unique(seg[rid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,1] = np.minimum(out[sid,1],rid)
        out[sid,2] = np.maximum(out[sid,2],rid)
    cids = np.where((seg>0).sum(axis=0)>0)[0]
    for cid in cids:
        sid = np.unique(seg[:,cid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,3] = np.minimum(out[sid,3],cid)
        out[sid,4] = np.maximum(out[sid,4],cid)

    if do_count:
        ui,uc = np.unique(seg,return_counts=True)
        out[ui,-1]=uc
    return out[uid]

def getSegType(mid):
    m_type = np.uint64
    if mid<2**8:
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def label_chunk(get_chunk, numC, rr=1, m_type=np.uint64):
    # need to implement get_chunk function
    # INPUT: chunk_id
    # OUTPUT: 3d chunk
    # label chunks or slices
    sz = get_chunk(0).shape
    numD = len(sz)
    
    mid = 0
    seg = [None]*numC
    for zi in range(numC):
        print('%d/%d [%d], '%(zi,numC,mid)),
        sys.stdout.flush()
        # as split as possible
        _, seg_c = cv2.connectedComponents(get_chunk(zi)>0, connectivity=4).astype(m_type)

        if numD==2:
            seg_c = seg_c[np.newaxis]

        if zi == 0: # first seg, relabel seg index        
            print('_%d_'%0)
            slice_b = seg_c[-1]
            seg[zi] = seg_c[:,::rr,::rr] # save a low-res one
            mid += seg[zi].max()
            rlA = np.arange(mid+1,dtype=m_type)
        else: # link to previous slice
            slice_t = seg_c[0]            
            _, slices = cv2.connectedComponents(np.stack([slice_b>0, slice_t>0],axis=0), connectivity=4).astype(m_type)
            # create mapping for seg cur
            lc = np.unique(seg_c);lc=lc[lc>0]
            rl_c = np.zeros(int(lc.max())+1, dtype=int)
            # merge curr seg
            # for 1 pre seg id -> slices id -> cur seg ids
            l0_p = np.unique(slice_b*(slices[0]>0))
            bbs = seg_bbox2d(slice_b, uid=l0_p)[:,1:] 
            print('_%d_'%len(l0_p))
            for i,l in enumerate(l0_p):
                bb = bbs[i]
                sid = np.unique(slices[0,bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(slice_b[bb[0]:bb[1]+1,bb[2]:bb[3]+1]==l))
                sid = sid[sid>0]
                # multiple ids
                if len(sid)==1:
                    cid = np.unique(slice_t*(slices[1]==sid))
                else:
                    cid = np.unique(slice_t*np.in1d(slices[1].reshape(-1),sid).reshape(sz[-2:]))
                rl_c[cid[cid>0]] = l
            
            # new id
            new_num = np.where(rl_c==0)[0][1:] # except the first one
            new_id = np.arange(mid+1,mid+1+len(new_num),dtype=m_type)
            rl_c[new_num] = new_id            
            slice_b = rl_c[seg_c[-1]] # save a high-res
            seg[zi] = rl_c[seg_c[:,::rr,::rr]]
            mid += len(new_num)
            
            # update global id
            rlA = np.hstack([rlA,new_id])
            # merge prev seg
            # for 1 cur seg id -> slices id -> prev seg ids
            l1_c = np.unique(slice_t*(slices[1]>0))
            for l in l1_c:
                sid = np.unique(slices[1]*(slice_t==l))
                sid = sid[sid>0]
                pid = np.unique(slice_b*np.in1d(slices[0].reshape(-1),sid).reshape(sz[-2:]))
                pid = pid[pid>0]
                # get all previous m-to-1 labels
                pid_p = np.where(np.in1d(rlA,rlA[pid]))[0]
                if len(pid_p)>1:
                    rlA[pid_p] = pid.max()
        # memory reduction: each seg
        m2_type = getSegType(seg[zi].max())
        seg[zi] = seg[zi].astype(m2_type)
    # memory reduction: final output
    m2_type = getSegType(rlA.max())
    rlA = rlA.astype(m2_type)
    print('output type:',m2_type)
    return rlA[np.vstack(seg)]

# 2. heatmap + seg -> detection score
def heatmap_to_score(seg, heatmap, channel=-1, do_avg=True):
    # 3D vol version
    if heatmap.ndim>seg.ndim:
        heatmap = heatmap_by_channel(heatmap, channel)

    seg_id, seg_count = np.unique(seg, return_counts=True)
    seg_view = seg.ravel()
    seg_len = seg_id.max()+1
    # relabel bincount(minlen = max_len) with ids
    score = np.bincount(seg_view, weights=heatmap.ravel(), minlength=seg_len)[seg_id]
    if do_avg:
        score = score/seg_count
        if score.max()>1: # assume 0-255
            score = score/255.
    return seg_id, score, seg_count

def heatmap_to_score_tile(seg_tiles, heatmap_tiles, max_id=-1, channel=-1):
    if max_id == -1:# rough estimate of the largest seg id
        max_id = max(100, 2*im2seg(cv2.imread(seg_tiles[-1])).max())
    count = np.zeros((max_id+1,2)) # num_voxel, sum_score
    for z in range(len(seg_tiles)):
        # 3D vol version
        seg = im2seg(cv2.imread(seg_tiles[z]))
        heatmap = cv2.imread(heatmap_tiles[z])
        t_id, t_score, t_count = heatmap_to_score(seg, heatmap, channel=-1, do_avg=False)
        # in case of wrong max_id input
        if t_id[-1]>max_id:
            out = np.vstack([out,np.zeros((max_id,2))])
            max_id *= 2
        count[t_id,0] += t_count
        count[t_id,1] += t_score

    pred_id = np.where(out[:,0]>0)[0]
    score = count[pred_id,1]/count[pred_id,0]
    if score.max()>1: # assume 0-255
        score = score/255.
    out = np.vstack([pred_id, score]).T 
    return out

# 3. instance seg -> bbox
def seg_bbox3d(seg,do_count=False, uid=None):
    """returns bounding box of segments"""
    sz = seg.shape
    assert len(sz)==3
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid>0]
    um = int(uid.max())
    out = np.zeros((1+um,7+do_count),dtype=np.uint32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1], out[:,3], out[:,5] = sz[0], sz[1], sz[2]

    # for each slice
    zids = np.where((seg>0).sum(axis=1).sum(axis=1)>0)[0]
    for zid in tqdm(zids):
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

def seg_bbox3d_tile(seg_tiles, do_count=False, max_id=-1):
    """returns bounding box of segments"""
    if max_id == -1:
        max_id = max(100, 2*im2seg(cv2.imread(seg_tiles[-1])).max())

    sz = cv2.imread(seg_tiles[0]).shape
    out = np.zeros((max_id+1, 7+do_count),dtype=np.uint32)
    out[:,1], out[:,3], out[:,5] = len(seg_tiles), sz[0], sz[1]

    # for each slice
    for z in tqdm(range(len(seg_tiles))):
        seg = cv2.imread(seg_tiles[z])
        if seg.max()>0:
            sid = np.unique(seg)
            sid = sid[sid>0]
            if sid[-1]>max_id:
                out = np.vstack([out,np.zeros((max_id, 7+do_count), np.uint32)])
                max_id = max_id*2
            # for z 
            out[sid,1] = np.minimum(out[sid,1],zid)
            out[sid,2] = np.maximum(out[sid,2],zid)

            # for each row
            rids = np.where((seg>0).sum(axis=1)>0)[0]
            for rid in rids:
                sid = np.unique(seg[rid])
                sid = sid[sid>0]
                if sid[-1]>max_id:
                    out = np.vstack([out,np.zeros((max_id, 7+do_count), np.uint32)])
                    max_id = max_id*2
                out[sid,3] = np.minimum(out[sid,3],rid)
                out[sid,4] = np.maximum(out[sid,4],rid)
            
            # for each col
            cids = np.where((seg>0).sum(axis=0)>0)[0]
            for cid in cids:
                sid = np.unique(seg[:,cid])
                sid = sid[sid>0]
                if sid[-1]>max_id:
                    out = np.vstack([out,np.zeros((max_id, 7+do_count), np.uint32)])
                    max_id = max_id*2
                out[sid,5] = np.minimum(out[sid,5],cid)
                out[sid,6] = np.maximum(out[sid,6],cid)

        if do_count:
            ui,uc = np.unique(seg,return_counts=True)
            out[ui[ui<=um],-1]=uc[ui<=um]

        out[:,0] = np.arange(max_id+1)
        return out[uid]


def seg_iou3d(pred, gt, areaRng, todo_id=None):
    # returns the matching pairs of ground truth IDs and prediction IDs, as well as the IoU of each pair.
    # (pred,gt)
    # return: id_1,id_2,size_1,size_2,iou
    pred_id, pred_sz = np.unique(pred, return_counts=True)
    pred_sz = pred_sz[pred_id>0]
    pred_id = pred_id[pred_id>0]
    predict_sz_rl = np.zeros(pred_id.max()+1,int)
    predict_sz_rl[pred_id] = pred_sz
    
    gt_id, gt_sz = np.unique(gt,return_counts=True)
    gt_sz=gt_sz[gt_id>0];gt_id=gt_id[gt_id>0]
    
    if todo_id is None:
        todo_id = pred_id
        todo_sz = pred_sz
    else:
        todo_sz = predict_sz_rl[todo_id]
   
    print('\t compute bounding boxes')
    bbs = seg_bbox3d(pred, uid=todo_id)[:,1:]    
    
    result_p = np.zeros((len(todo_id), 2+3*areaRng.shape[0]), float)
    result_p[:,0] = todo_id
    result_p[:,1] = todo_sz

    gt_matched_id = np.zeros(1+gt_id.max(), int)
    gt_matched_iou = np.zeros(1+gt_id.max(), float)

    print('\t compute iou matching')
    for j,i in tqdm(enumerate(todo_id)):
        # Find intersection of pred and gt instance inside bbox, call intersection match_id
        bb = bbs[j]
        match_id, match_sz=np.unique(gt[bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(pred[bb[0]:bb[1]+1,bb[2]:bb[3]+1]==i),return_counts=True)
        match_sz=match_sz[match_id>0] # get intersection counts
        match_id=match_id[match_id>0] # get intersection ids        
        if len(match_id)>0:
            # get count of all preds inside bbox (assume gt_id,match_id are of ascending order)
            gt_sz_match = gt_sz[np.isin(gt_id, match_id)]
            ious = match_sz.astype(float)/(todo_sz[j] + gt_sz_match - match_sz) #all possible iou combinations of bbox ids are contained
            
            for r in range(areaRng.shape[0]): # fill up all, then s, m, l
                gid = (gt_sz_match>areaRng[r,0])*(gt_sz_match<=areaRng[r,1])
                if sum(gid)>0: 
                    idx_iou_max = np.argmax(ious*gid)
                    result_p[j,2+r*3:2+r*3+3] = [ match_id[idx_iou_max], gt_sz_match[idx_iou_max], ious[idx_iou_max] ]            
            # update set2
            gt_todo = gt_matched_iou[match_id]<ious            
            gt_matched_iou[match_id[gt_todo]] = ious[gt_todo]
            gt_matched_id[match_id[gt_todo]] = i
                
    # get the rest: false negative + dup
    fn_gid = gt_id[np.isin(gt_id, result_p[:,2], assume_unique=False, invert=True)]
    fn_gic = gt_sz[np.isin(gt_id, fn_gid)]
    fn_iou = gt_matched_iou[fn_gid]
    fn_pid = gt_matched_id[fn_gid]
    fn_pic = predict_sz_rl[fn_pid]
    
    # add back duplicate
    # instead of bookkeeping in the previous step, faster to redo them    
    result_fn = np.vstack([fn_pid, fn_pic, fn_gid, fn_gic, fn_iou]).T
    
    return result_p, result_fn

def seg_iou3d_sorted(pred, gt, score, areaRng=[0,1e10]):
    # pred_score: Nx2 [id, score]
    # 1. sort prediction by confidence score
    relabel = np.zeros(int(np.max(score[:,0])+1), float)
    relabel[score[:,0].astype(int)] = score[:,1]
    
    # 1. sort the prediction by confidence
    pred_id = np.unique(pred)
    pred_id = pred_id[pred_id>0]
    pred_id_sorted = np.argsort(-relabel[pred_id])
    
    result_p, result_fn = seg_iou3d(pred, gt, areaRng, todo_id=pred_id[pred_id_sorted])
    # format: pid,pc,p_score, gid,gc,iou
    pred_score_sorted = relabel[pred_id_sorted].reshape(-1,1)
    return result_p, result_fn, pred_score_sorted
