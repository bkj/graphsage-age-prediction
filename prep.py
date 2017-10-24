#!/usr/bin/env python

"""
    prep.py
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import networkx as nx

np.random.seed(123)

# --
# Helpers

def load_ages(path):
    ages = pd.read_csv(path, header=None, sep='\t')
    ages.columns = ('id', 'age')
    
    ages = ages[ages.age != 'null']
    ages.age = ages.age.astype(int)
    ages = ages[ages.age > 0]
    
    return ages


def make_adjacency(G, folds, max_degree, train=True):
    
    all_nodes = np.array(G.nodes())
    
    # Initialize w/ links to a dummy node
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes + 1, max_degree)) + n_nodes).astype(int)
    
    if train:
        # only look at nodes in training set
        all_nodes = all_nodes[folds == 'train']
    
    for node in all_nodes:
        neibs = np.array(G.neighbors(node))
        
        if train:
            neibs = neibs[folds[neibs] == 'train']
        
        if len(neibs) > 0:
            if len(neibs) > max_degree:
                neibs = np.random.choice(neibs, max_degree, replace=False)
            elif len(neibs) < max_degree:
                neibs = np.random.choice(neibs, max_degree, replace=True)
            
            adj[node, :] = neibs
    
    return adj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='../data/pokec/')
    parser.add_argument('--outpath', type=str, default='../data/pokec/problem.h5')
    parser.add_argument('--max-degree', type=int, default=128)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    # --
    # Load data
    
    ages  = load_ages(os.path.join(args.inpath, 'soc-pokec-ages.tsv'))
    edges = pd.read_csv(os.path.join(args.inpath, 'soc-pokec-relationships.txt'), header=None, sep='\t')
    edges.columns = ('src', 'trg')
    
    # Remove orphans, etc
    edges = edges[edges.src.isin(ages.id)]
    edges = edges[edges.trg.isin(ages.id)]
    ages  = ages[ages.id.isin(edges.src) | ages.id.isin(edges.trg)]
    
    # Create unique id
    ages['uid'] = np.arange(ages.shape[0])
    
    # Use 
    edges = pd.merge(edges, ages, left_on='src', right_on='id')
    edges = edges[['uid', 'trg']]
    edges.columns = ('src', 'trg')
    edges = pd.merge(edges, ages, left_on='trg', right_on='id')
    edges = edges[['src', 'uid']]
    edges.columns = ('src', 'trg')
    
    ages = ages[['uid', 'age']]
    
    # --
    # Format data
    
    targets = np.array(ages.age).astype(float).reshape(-1, 1)
    folds = np.random.choice(['train', 'val'], targets.shape[0], p=[0.8, 0.2])
    
    G = nx.from_edgelist(np.array(edges))
    adj = make_adjacency(G, folds, args.max_degree, train=False) # Adds dummy node
    
    # --
    # Write data
    
    problem = {
        "adj" : adj,
        "targets" : targets,
        "folds" : folds,
    }
    
    f = h5py.File(args.outpath)
    for k,v in problem.items():
        f[k] = v
        
    f.close()