import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch


def encode(labels):
    c=[]
    for e in labels :
        if not (e==0 or e==str(0)):
            c.append(str(e))
    classes=[]
    for e in c:
        if e not in classes:
            classes.append(e)
    class_dict={str(i):np.identity(len(classes))[i-1,:] for i in range(1,len(classes)+1)}
    labels_OH=[]
    for i in range(len(labels)) :
        e=labels[i]
        if e==0 or e==str(0):
            labels_OH.append(list(np.zeros(len(classes))))
        else : 
            labels_OH.append(list(class_dict.get(str(e))))
    return labels_OH




def load_data(path1, path2): #path1 = document des co-auteurs(.csv); path2 = documents des vecteurs(.txt)
    """Load citation network dataset (cora only for now)"""
    print('Loading dataset...')

    dfauth=pd.read_csv(path1)


    idx_features_labels = np.genfromtxt(path2,dtype=np.dtype(str)) #sans la colonne des labels
    features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)
    lab=list(dfauth['Classes'])

    labels=np.array(encode(lab))

    # build graph
    idx = np.array(dfauth['Id'])
    idx_map = {j: i for i, j in enumerate(idx)}

    #edges_unordered est une liste d'array regroupant des couples d'id de coauteurs : [id_coauteur, id_auteur]

    edges_unordered=[]

    L=list(dfauth['Id'])


    for index, row in dfauth.iterrows():
        if isinstance(row['Co_auth'],str):
          co_auth=row['Co_auth'].split(',') #liste des co-auteurs pour un auteur donné
          for c in co_auth:
            if int(c) in L:
              sub_arr=[]
              sub_arr.append(row['Id'])
              sub_arr.append(int(c))
              edges_unordered.append(sub_arr)
    
    edges_unord=np.array(edges_unordered).flatten() 

    edges = np.array(list(map(idx_map.get, edges_unord)),
                     dtype=np.int32).reshape(np.array(edges_unordered).shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    
    idx_train = []
    C=list(dfauth['Classes'])
    print('***')
    for i in range(len(C)):
      if not C[i]==0:
        idx_train.append(i)

    idx_f=range(len(idx_train))



    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_f=torch.LongTensor(idx_f)

    return adj, features, labels, idx_train,idx_f

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)