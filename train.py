#!/usr/bin/env python

"""
    train.py
"""

from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np

from models import GSSupervised
from problem import NodeProblem
from helpers import set_seeds, to_numpy

# --
# Helpers

def evaluate(model, problem):
    preds, acts = [], []
    for ids, targets, _ in problem.iterate(mode='val', shuffle=False):
        preds.append(to_numpy(model(ids, problem.adj, train=True)))
        acts.append(to_numpy(targets))
    
    return problem.metric_fn(np.vstack(acts), np.vstack(preds))

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--problem-path', type=str, required=True)
    parser.add_argument('--no-cuda', action="store_true")
    parser.add_argument('--seed', default=123, type=int)
    
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr-init', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    
    # --
    # Validate args
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert args.batch_size > 1, 'parse_args: batch_size must be > 1'
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # Load problem
    
    problem = NodeProblem(problem_path=args.problem_path, cuda=args.cuda)
    
    # --
    # Define model
    
    model = GSSupervised(**{
        "n_nodes"      : problem.n_nodes,
        "lr_init"      : args.lr_init,
        "weight_decay" : args.weight_decay,
    })
    
    if args.cuda:
        model = model.cuda()
    
    print(model, file=sys.stderr)
    
    # --
    # Train
    
    set_seeds(args.seed ** 2)
    
    for epoch in range(args.epochs):
        
        # Train
        _ = model.train()
        for ids, targets, progress in problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size):
            preds = model.train_step(
                ids=ids, 
                adj=problem.adj,
                targets=targets,
                loss_fn=problem.loss_fn,
            )
            
            sys.stderr.write("\repoch=%d | progress=%f" % (epoch, progress))
            sys.stderr.flush()
        
        # Evaluate
        _ = model.eval()
        print()
        print({
            "epoch" : epoch,
            "train_metric" : problem.metric_fn(to_numpy(targets), to_numpy(preds)),
            "val_metric" : evaluate(model, problem),
        })
        print()
