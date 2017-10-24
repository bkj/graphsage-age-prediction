#!/usr/bin/env python

"""
    models.py
"""

from __future__ import division
from __future__ import print_function

from functools import partial

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# --
# Helpers

def uniform_neighbor_sampler(ids, adj, n_samples=-1):
    tmp = adj[ids]
    perm = torch.randperm(tmp.size(1))
    if adj.is_cuda:
        perm = perm.cuda()
    
    tmp = tmp[:,perm]
    return tmp[:,:n_samples]


# --
# Preprocessers

class NodeEmbeddingPrep(nn.Module):
    def __init__(self, n_nodes, embedding_dim=64):
        super(NodeEmbeddingPrep, self).__init__()
        
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes + 1, embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim) # Affine transform, for changing scale + location
    
    @property
    def output_dim(self):
        return self.embedding_dim
    
    def forward(self, ids, adj, layer_idx=0):
        if layer_idx > 0:
            embs = self.embedding(ids)
        else:
            # Don't look at node's own embedding for prediction, or you'll probably overfit a lot
            embs = self.embedding(Variable(ids.clone().data.zero_() + self.n_nodes))
        
        return self.fc(embs)


# --
# Aggregators 

class MeanAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, activation, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanAggregator, self).__init__()
        
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn
    
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)
        
    def forward(self, x, neibs):
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1)) # !! Careful
        agg_neib = agg_neib.mean(dim=1) # Careful
        
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        
        return out



# --
# Model

class GSSupervised(nn.Module):
    def __init__(self, n_nodes, lr_init=0.01, weight_decay=0.0):
        super(GSSupervised, self).__init__()
        
        layer_specs = [
            {
                "sample_fn" : uniform_neighbor_sampler,
                "n_samples" : 25,
                "output_dim" : 128,
                "activation" : F.relu,
            },
            {
                "sample_fn" : uniform_neighbor_sampler,
                "n_samples" : 25,
                "output_dim" : 128,
                "activation" : lambda x: x,
            },
        ]
        
        # --
        # Define network
        self.prep = NodeEmbeddingPrep(n_nodes=n_nodes)
        input_dim = self.prep.output_dim
        agg_layers = []
        for spec in layer_specs:
            agg = MeanAggregator(
                input_dim=input_dim,
                output_dim=spec['output_dim'],
                activation=spec['activation'],
            )
            agg_layers.append(agg)
            input_dim = agg.output_dim # May not be the same as spec['output_dim']
        
        self.agg_layers = nn.Sequential(*agg_layers)
        self.fc = nn.Linear(input_dim, 1, bias=True)
        
        # --
        # Setup samplers
        
        self.sample_fns = [partial(s['sample_fn'], n_samples=s['n_samples']) for s in layer_specs]
        
        # --
        # Define optimizer
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_init, weight_decay=weight_decay)
    
    def _sample(self, ids, adj, train):
        all_feats = [self.prep(ids, adj, layer_idx=0)]
        for layer_idx, sampler_fn in enumerate(self.sample_fns):
            ids = sampler_fn(ids=ids, adj=adj).contiguous().view(-1)
            all_feats.append(self.prep(ids, adj, layer_idx=layer_idx + 1))
        
        return all_feats
    
    def forward(self, ids, adj, train=True):
        # Sample neighbors + apply `prep_class`
        all_feats = self._sample(ids, adj, train=train)
        
        # Sequentially apply layers, per original (little weird, IMO)
        # Each iteration reduces length of array by one
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
        
        assert len(all_feats) == 1, "len(all_feats) != 1"
        
        out = F.normalize(all_feats[0], dim=1) # ??
        return self.fc(out)
    
    def train_step(self, ids, adj, targets, loss_fn):
        self.optimizer.zero_grad()
        preds = self(ids, adj)
        loss = loss_fn(preds, targets.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return preds

