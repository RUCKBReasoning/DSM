import numpy as np
import torch

def load_data(datasets):
    # load the adjacency
    adj = np.loadtxt('./data/'+datasets+'.txt')
    adj = adj.astype('int')
    nb_nodes = np.max(adj) + 1
    edge_index = adj.T
    print('Load the edge_index done!')    
   
    # load initial features
    feats = np.load('./features/'+datasets+'_feature1024.npy')
    return edge_index, feats, nb_nodes


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.len = len(inputs)
        self.data = inputs
    def __getitem__(self, index):              
        return self.data[index]
    def __len__(self):
        return self.len
