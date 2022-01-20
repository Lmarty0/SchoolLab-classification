import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCN(nn.Module):
    
    
    def __init__(self, nfeat, nhid, nclass, dropout,nlayers): #L'argument nlayers a été ajouté

        super(GCN, self).__init__()
        
        self.layers=[0 for i in range(nlayers)]
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        
        #Création d'une liste de nlayers couches
        self.layers[0]=self.gc1
        self.layers[-1]=self.gc2
        for i in range(1,nlayers-1):
            self.g=GraphConvolution(nhid, nhid)
            self.layers[i]=self.g
        	
        self.dropout = dropout

    def forward(self, x, adj):
        
        nl=len(self.layers)        
        for i in range(nl-1):
                gc=self.layers[i]
                x = F.relu(gc(x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)