import numpy as np
import json
import pickle
import torch
import os
from random import choice
from bert_embedding import BertEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import sys
import random
import  argparse
np.random.seed(222)

# generate graph adj
def process_graph_adj(mode, input_dir):
    filePath = os.path.join(input_dir, mode + '.json')    
    with open(filePath) as f:
        ind = 0
        adjList = []
        nodeIdToNameIndex = dict()
        for line in f.readlines():
            line = json.loads(line)
            nodeIdNameDict = dict()            
            g_node_names = line['inGraph']['g_node_names']
            for key, _ in g_node_names.items():
                nodeIdNameDict[key] = ind
                nodeIdToNameIndex[ind] = key              
                ind += 1
            g_adj = line['inGraph']['g_adj']
            for key, value in g_adj.items():                
                for element in list(value.keys()):
                    adjElement = []
                    adjElement.append(nodeIdNameDict[key])
                    adjElement.append(nodeIdNameDict[element])
                    adjList.append(adjElement)
   
    np.savetxt('./data/WQ' + mode +'Graph.txt', adjList, fmt = '%d') 
   
# process the graph initial feature
def process_feature(mode, input_dir):
    filePath = os.path.join(input_dir, mode + '.json')
    nodeEmbeddingList = []# all node embedding's set, every element node embedding is a list 
    bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
    with open(filePath) as f1:
        for line in f1.readlines():
            nodeList = []
            res = []
            embDict = dict()
            line = json.loads(line)
            relation = list(line['inGraph']['g_edge_types'].values())
            for r in relation:
                r = r.split('/')[-1]
                res.append(r)
            res.append('[MASK]')            
            result = bert_embedding(res) #result is a list, the element of list is a tuple, include  (token, token embedding)
            # print(res, type(res))
            for (_, emb), re in zip(result, res): #emb is a list, the embedding is a 1*1024 dimension
                emb = np.array(emb)
                emb = np.sum(emb, axis=0)
                embDict[re] = emb
            g_node_names = line['inGraph']['g_node_names']
            for key, _ in g_node_names.items():
                nodeList.append(key) # nodeList is the subgraph node list
            g_adj = line['inGraph']['g_adj']
            g_adj_value = list(g_adj.values()) # list element is a dict

            # deal with g_adj_value to a dict, the dict element is {node:[relation list]}
            nodeToRelationdict = dict()
            for nodeToRelation in g_adj_value:
                for k, v in nodeToRelation.items():
                    if k in list(nodeToRelationdict.keys()):
                        nodeToRelationdict[k].append(v)
                    else:
                        nodeToRelationdict[k] = []
                        nodeToRelationdict[k].append(v)

            for node in nodeList:
                relationEmb = []
                if node in list(nodeToRelationdict.keys()):
                    for r in nodeToRelationdict[node]:
                        r = r.split('/')[-1]
                        relationEmb.append(embDict[r])
                    relationEmb = np.mean(np.array(relationEmb), axis = 0)
                    nodeEmbeddingList.append(relationEmb)
                else:
                    nodeEmbeddingList.append(embDict['[MASK]'])
               
    nodeEmbeddingList = np.array(nodeEmbeddingList) 
    np.save('./data/WQTrainGraph_feature1024.npy', nodeEmbeddingList)
    print(len(nodeEmbeddingList))
 
# generate each subgraph nodeID set
def process_graph_subNode(mode, input_dir):
    filePath = os.path.join(input_dir, mode + '.json')    
    with open(filePath) as f:
        ind = 0
        nodeList = []
        for line in f.readlines():
            line = json.loads(line)          
            g_node_names = line['inGraph']['g_node_names']
            temp_node = []
            for _ in list(g_node_names.keys()):                
                temp_node.append(ind)            
                ind += 1
            nodeList.append(temp_node)
    
    np.save('./data/WQ'+ mode + 'GraphNodeSet.npy', nodeList) 

        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_dir', required=True)
    argparser.add_argument('--mode', default = 'train')
    args = argparser.parse_args()

    #generate adj
    process_graph_adj(args.mode, args.input_dir)
          
    #generate graph initial feature
    process_feature(args.mode, args.input_dir)

    #generate subgraph node set
    process_graph_subNode(args.mode, args.input_dir)

    
