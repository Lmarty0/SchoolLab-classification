from __future__ import division
from __future__ import print_function

import scipy.stats

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_imt import load_data
from models import GCN

path1="C:/Users/lauri/Documents/Projet Recherche/Graphe IMT/gcn/Rendu Final/Data_set/co_auths_bis3.csv"
path2="C:/Users/lauri/Documents/Projet Recherche/Graphe IMT/gcn/Rendu Final/Data_set/data_set_vf.txt"

# Training settings

seed=42
epochs=200
lr=0.01
weight_decay=5e-4
hidden=16
dropout=0.5


np.random.seed(seed)
torch.manual_seed(seed)
adj, features, labels, idx_train, idx_f = load_data(path1,path2)

def train(epoch):

  t = time.time()
  model.train()
  optimizer.zero_grad()
  output = model(features, adj)
  loss_train = F.nll_loss(output[idx_train], labels[idx_f])
  loss_train.backward()
  optimizer.step()  

    
# Model and optimizer   
def initialization_model(nl,nf):
    global features, labels, dropout, Nlayers, model, optimizer

    model = GCN(nfeat=features.shape[1],
                nhid=nf,		#args.hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout,
                nlayers=nl)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    global adj, idx_train 
    #idx_val, idx_test


#Training all the epochs
def train_epochs(Nepochs):
    t_total = time.time()
    for epoch in range(Nepochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


def full_train():
  Nepochs=epochs #Already define in the starting parameters of the algorithm
  initialization_model(2,hidden)
  train_epochs(Nepochs)
  "/n/n/n"
  print("FINALLY")
  return model