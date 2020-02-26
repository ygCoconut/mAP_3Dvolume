__author__ = 'ychfan'

import numpy as np
import datetime
import time
from collections import defaultdict
# from . import mask as maskUtils
import copy

class YTVOSeval:
    # Interface for evaluating video instance segmentation on the YouTubeVIS dataset.
    #
    # The usage for YTVOSeval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = YTVOSeval(cocoGt,cocoDt); # initialize YTVOSeval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, ID_map, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        num_rows = ID_map.shape[0]
        self.ID_map = ID_map
        self.cocoGt = ID_map[:,:2]    # ground truth COCO API
        self.cocoDt = ID_map[:,3:].reshape((num_rows, -1, 3)) # detections COCO API
        
        self.ious = self.cocoDt[:,:,-1] # contains [all, large, medium, small]
#         self.scores = ID_map[:,2] if 0.0 <= number <= 1.0 else ID_map[:,2]/255.0
        
#         self.fps = np.zeros(self.cocoDt.shape[1])
#         self.fns = np.zeros_like(self.fps)
#         self.tps = np.zeros_like(self.fps)
#         self.fps[0] = np.count_nonzero(self.cocoGt[:,0] == 0) #trick to count zero values
#         self.fns[0] = np.count_nonzero(ID_map[:,3] == 0)
#         self.tps[0] = num_rows - self.fns - self.fps
        
#         self.fps = np.count_nonzero(self.cocoGt[:,0] == 0) #trick to count zero values
#         self.fns = np.count_nonzero(ID_map[:,3] == 0)
#         self.tps = num_rows - self.fns - self.fps
   
        self.params   = {}                  # evaluation parameters
        self.eval     = {}                  # accumulated evaluation results
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self._paramsEval = copy.deepcopy(self.params)
        self.stats = []                     # result summarization
#         self.ious = {}                      # ious between all gts and dts
#         if not cocoGt is None:
        self.params.vidIds = [0]
        self.params.catIds = [1]
            
            
    def get_tfpn(self):
        """
        For each instance, we need the number of true positives, false positives and false negatives
        at each IoU threshold.
        """        
        T = len(self.params.iouThrs)
        D = self.cocoDt.shape[0]
        
        self.fps = np.zeros((T, D))
        self.fns = np.zeros_like(self.fps)
        self.tps = np.zeros_like(self.fps)
        
        # masks to isolate True values
        fps_mask = self.cocoGt[:,0] == 0 #fps have nonzero IoUs for all the ranges, TODO: add dim for range
        fns_mask = self.ID_map[:,3] == 0
        tps_mask = ~(fps_mask + fns_mask)
        
        
        th = self.params.iouThrs.repeat(D).reshape((T, -1)) #get same length as ious
        tps = (self.ious[:,0]*tps_mask > th) #fps have nonzero IoUs for all the ranges, TODO: add dim for range
        fps = fps_mask.repeat(T).reshape((-1, T)).T
        fns = fns_mask.repeat(T).reshape((-1, T)).T + ~(tps + fps)
        
        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        scores = self.ID_map[:,2][tps_mask]#/255.0
        
        inds = np.argsort(-scores, kind='mergesort')
        self.scores = scores[inds]
        
        #### depends on previously correct indexing.
        self.tps = tps[:,tps_mask][:,inds]
        self.fps = ~tps[:,tps_mask][:,inds]
        self.fns = fns[:,fns_mask]
        
        self.inds = inds
        
    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''

        print('Accumulating evaluation results...')
        tic = time.time()
#         if not self.evalImgs:
#             print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))
        
        # create dictionary for future indexing
        _pe = self.params
#         _pe = self._paramsEval
#         _pe.useCats = 1

#         catIds = [1]
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.vidIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.vidIds)  if i in setI]
        I0 = len(_pe.vidIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
#                     E = [self.evalImgs[Nk + Na + i] for i in i_list]
#                     E = [e for e in E if not e is None]
#                     if len(E) == 0:
#                         continue
                    


                    # Get fp, fn and tp
                    self.get_tfpn()

            
            
#                     dtScores = self.scores
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
#                     inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = self.scores
                    
                    tps = self.tps
                    fps = self.fps
                    npig = tps.shape[1] + self.fns.shape[1]
                    
                    
                    # Sort tp, fp and scores in accuracy order
#                     npig = np.sum(tps +self.fns)/tps.shape[0]
#                     npig = 38
                    
#                     dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

#                     # different sorting method generates slightly different results.
#                     # mergesort is used to be consistent as Matlab implementation.
#                     inds = np.argsort(-dtScores, kind='mergesort')
#                     dtScoresSorted = dtScores[inds]

#                     dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
#                     dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
#                     gtIg = np.concatenate([e['gtIgnore'] for e in E])
#                     npig = np.count_nonzero(gtIg==0 )
#                     if npig == 0:
#                         continue
#                     tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
#                     fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    
                    import pdb; pdb.set_trace()        
                    
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
                        
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
#             'counts': [T, R, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
#                 s = s[:,:,aind,mind] #took out cat dim
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
#                 s = s[:,:,aind,mind]# took out cat dim
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets

        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.vidIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 128 ** 2], [ 128 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1


    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
