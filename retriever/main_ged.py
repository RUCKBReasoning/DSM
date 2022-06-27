from torch.multiprocessing import Pool
import multiprocessing as mp
import torch
import numpy as np
import time
import networkx as nx
import heapq
import json
import random
import os
import sys
import pickle

def getKG(input_dir, mode):
    trainPath = os.path.join(input_dir, mode +'.json')
    IndexList = []# test's index set
    kbList = [] # all sub-kb's set, every element is a sub_kb
    
    with open(trainPath) as f1:
        for line in f1.readlines():
            line = json.loads(line)
            g_adj = line['inGraph']['g_adj'] #g_adj is a dict list
            temp_kb_edge = []
            for k, v in g_adj.items():
                v_node = list(v.keys())
                for vv in v_node:
                    tx = []
                    tx.append(k)
                    tx.append(vv)
                    temp_kb_edge.append(tuple(tx))
            G = nx.DiGraph()
            G.add_edges_from(temp_kb_edge)
            kbList.append(G) #sub_kb is completed     
    return kbList       

def edit_distance_train(kbList, data):  
       
    l = len(kbList)
    editScore = []
    for d in data:
        editScoreList = []
        idx = kbList.index(d)
        editScoreList.append(idx)
        for i in range(l):                
            edit_distance = 12345
            for v in nx.optimize_graph_edit_distance(d, kbList[i], upper_bound = 50):  # optimized edit distance
                if v < edit_distance:
                    edit_distance = v  
            editScoreList.append(edit_distance)
        editScore.append(editScoreList)
    np.save("./multiProcess_GED_PQ_Train/PQ_multiTrain_"+ str(time.time()) +"_editscore.npy", editScore)


def multiprocess_train(input_dir):
    p = Pool(60) # original 30
    pos_data = getKG(input_dir, 'train')
    each_p_num = int(len(pos_data)/60)
    for i in range(0,len(pos_data),each_p_num):
        p.apply_async(func=edit_distance_train, args=(pos_data, pos_data[i:i+each_p_num], )) #!!!!!args need ,
    p.close()
    p.join()

################### multiTest ##################################################
def edit_distance_test(kbList, test_kbList, data, k_shot):
    l = len(kbList)
    IndexList = []
    for d in data:
        index = test_kbList.index(d)
        k_shot_Index = [] # top k_shot revelance index in raw training data
        k_shot_Index.append(index)
        query_kb = test_kbList[index]
        scoreDict = dict() 
        for i in range(l):
            edit_distance = 12345
            for v in nx.optimize_graph_edit_distance(query_kb, kbList[i], upper_bound = 50):  # optimized edit distance
                if v < edit_distance:
                    edit_distance = v
            scoreDict[i] = edit_distance
        scoreList = sorted(scoreDict.items(),key = lambda x:x[1],reverse = True)
        scoreList = scoreList[:k_shot]
        for ind, _ in scoreList:
            k_shot_Index.append(ind)
        IndexList.append(k_shot_Index)
    np.savetxt("./multiProcess_GED_WQ_Test/WQ_multiTest_"+ str(time.time())+ "_edit.txt", IndexList, fmt="%d")

def multiprocess_test(input_dir):
    p = Pool(40)  # original is 30
    kbList = getKG(input_dir, 'train')
    k_shot = 20
    pos_data = getKG(input_dir, 'test')
    each_p_num = int(len(pos_data)/40)
    for i in range(0,len(pos_data),each_p_num):
        p.apply_async(func=edit_distance_test, args=(kbList, pos_data, pos_data[i:i+each_p_num], k_shot, )) #!!!!!args need ,
    p.close()
    p.join()
    
if __name__=='__main__':
    input_dir = './dataset/WQ'
    # multiprocess_train(input_dir) # multi_train
    multiprocess_test(input_dir) # multi_test
  

    


