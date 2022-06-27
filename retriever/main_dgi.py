import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
import random

from util import load_data
from dgi import DGI
import sys

sig = torch.nn.Sigmoid()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  

def preprocess_neighbors_sumavepool(edge_index, nb_nodes, device):
    adj_idx = edge_index
        
    self_loop_edge = torch.LongTensor([range(nb_nodes), range(nb_nodes)])
    adj_idx = torch.cat([adj_idx, self_loop_edge], 1)
        
    adj_elem = torch.ones(adj_idx.shape[1])

    adj = torch.sparse.FloatTensor(adj_idx, adj_elem, torch.Size([nb_nodes, nb_nodes]))

    return adj.to(device)


def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net')
    parser.add_argument('--dataset', type=str, default="WQTrainGraph",
                        help='name of dataset (default: WQTrainGraph/WQTestGraph)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers (default: 2)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='number of hidden units (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over neighboring nodes: sum or average')
    args = parser.parse_args()

    setup_seed(0)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # Data loading
    edge_index, feats, nb_nodes = load_data(args.dataset)
    input_dim = feats.shape[1]
    # the shuffled features are used to contruct the negative samples
    idx = np.random.permutation(nb_nodes)
    shuf_feats = feats[idx, :]

    model = DGI(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, args.neighbor_pooling_type, device).to(device)
    optimizer_train = optim.Adam(model.parameters(), lr=args.lr)

    batch_size = 1
    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

    adj = preprocess_neighbors_sumavepool(torch.LongTensor(edge_index), nb_nodes, device)
    feats = torch.FloatTensor(feats).to(device)
    shuf_feats = torch.FloatTensor(shuf_feats).to(device)
    
    cnt_wait = 0
    best = 1e9
    best_t = 1
    patience = 15

    # training
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_pretrain = model(feats, shuf_feats, adj, None, None, None, lbl)
        print("loss:", loss_pretrain.item())
        if epoch > 10:
            if loss_pretrain.item() < best:
                best = loss_pretrain.item()
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), './best_DGI_param_WQ.pt', _use_new_zipfile_serialization = False) 
            else:
                cnt_wait +=1
        
        if cnt_wait == patience:
            print("Early Stopping!, epoch:", epoch)
            break

        if optimizer_train is not None:
            optimizer_train.zero_grad()
            loss_pretrain.backward()         
            optimizer_train.step()
     
    #load model 
    model.load_state_dict(torch.load('./best_DGI_param_WQ.pt'))  
    emb = model.get_emb(feats, adj)
    np.save('./DGI_GraphData/' + args.dataset +'_node1024.npy', emb)
    print('Training done!')

    # calculate subgraph embedding
    train_nodeSet = np.load('./data/WQTrainGraphNodeSet.npy', allow_pickle=True)
    trainSubGraphEmd = []
    for index in train_nodeSet:    
        trainSubGraphEmd.append(np.mean(emd[index], axis = 0))

    np.save('./wholeSubgraphEmb/WQTrainSubGraph_DGI_emb1024', np.array(trainSubGraphEmd))

if __name__ == '__main__':
    main()
