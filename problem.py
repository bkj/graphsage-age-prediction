#!/usr/bin/env python

"""
    problem.py
"""

from __future__ import division
from __future__ import print_function

import h5py
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import functional as F

# --
# Helper classes

def loss_fn(preds, targets):
    return F.l1_loss(preds, targets)

def corr(a, b):
    a -= a.mean()
    b -= b.mean()
    return np.mean(a * b) / (np.std(a) * np.std(b))

def metric_fn(y_true, y_pred):
    return {
        "mae" : np.abs(y_true - y_pred).mean(),
        "corr" : corr(y_true, y_pred),
    }

# --
# Problem definition

class NodeProblem(object):
    def __init__(self, problem_path, cuda=True):
        
        print('NodeProblem: loading started')
        
        f = h5py.File(problem_path)
        self.n_classes = f['n_classes'].value if 'n_classes' in f else 1 # !!
        self.folds     = f['folds'].value
        self.targets   = f['targets'].value
        self.adj       = f['adj'].value
        f.close()
        
        self.n_nodes   = self.adj.shape[0]
        self.cuda      = cuda
        self.__to_torch()
        
        self.nodes = {
            "train" : np.where(self.folds == 'train')[0],
            "val"   : np.where(self.folds == 'val')[0],
        }
        
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        
        print('NodeProblem: loading finished')
    
    def __to_torch(self):
        """ convert adj to torch """
        self.adj = Variable(torch.LongTensor(self.adj))
        
        if self.cuda:
            self.adj = self.adj.cuda()
    
    def __batch_to_torch(self, mids, targets):
        """ convert batch to torch """
        mids = Variable(torch.LongTensor(mids))
        targets = Variable(torch.FloatTensor(targets))
        
        if self.cuda:
            mids, targets = mids.cuda(), targets.cuda()
        
        return mids, targets
    
    def iterate(self, mode, batch_size=512, shuffle=False):
        nodes = self.nodes[mode]
        
        idx = np.arange(nodes.shape[0])
        if shuffle:
            idx = np.random.permutation(idx)
        
        n_chunks = idx.shape[0] // batch_size + 1
        for chunk_idx, chunk in enumerate(np.array_split(idx, n_chunks)):
            mids = nodes[chunk]
            targets = self.targets[mids]
            mids, targets = self.__batch_to_torch(mids, targets)
            yield mids, targets, chunk_idx / n_chunks
