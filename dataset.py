import json
import pickle
import torch
import os
import numpy as np
from random import choice
from bert_embedding import BertEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
np.random.seed(222)

def collate(batch):
    batch = list(zip(*batch))
    support_x = torch.stack(batch[0])
    query_x = torch.stack(batch[1])
    query_x_index = torch.stack(batch[2])
    return support_x, query_x, query_x_index

#subgraph similarity
def relevance(index, mode, input_dir, k_shot):
    testGraphEmb = np.load('./wholeSubgraphEmb/WQTestSubGraph_CL_20_emb1024.npy')
    trainGraphEmb = np.load('./wholeSubgraphEmb/WQTrainSubGraph_CL_20_emb1024.npy')
    if mode =='train':
        query_re = trainGraphEmb[index]
        query_re = query_re[np.newaxis, :]
        similarity = cosine_similarity(query_re, trainGraphEmb)#score is an array
        similarity = np.squeeze(similarity, axis = 0)
        similarity = np.delete(similarity, index)
        index_number = np.argsort(-similarity)
        k_shot_Index = index_number[:k_shot]
    if mode =='test':
        query_re = testGraphEmb[index]
        query_re = query_re[np.newaxis, :]
        similarity = cosine_similarity(query_re, trainGraphEmb)#score is an array
        similarity = np.squeeze(similarity, axis = 0)
        index_number = np.argsort(-similarity)
        k_shot_Index = index_number[:k_shot]  
    return k_shot_Index


class Dataset(torch.utils.data.Dataset):
    def __init__(self, question_pt, input_dir, output_dir, mode, taskNum, k_shot, k_query=1):
      
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(4):
                inputs.append(pickle.load(f)) 
        self.source_ids, self.source_mask, self.target_ids, self.answers = inputs
        self.taskNum = taskNum
        self.k_shot = k_shot
        self.k_query = k_query
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mode = mode
        self.create_batch(self.taskNum)

    def create_batch(self, taskNum): # create  taskNum task
        self.support_x_batch = []  # support set task
        self.query_x_batch = []  # query set task
        self.query_x_index = [] # query set in raw train/test index, it is used to calculate BLEU score
        if self.mode == 'train':
            for i in range(taskNum):  # for each task
                k_shot_Index = relevance(i, self.mode, self.input_dir, self.k_shot)
                support_x = [] # n_way list, per list is a class list 
                query_x = []
                source_ids_temp = []
                source_mask_temp = []
                target_ids_temp = []
                answers_temp = []
                for cls in k_shot_Index:
                    source_ids_temp.append(self.source_ids[cls])
                    source_mask_temp.append(self.source_mask[cls])
                    target_ids_temp.append(self.target_ids[cls])
                    answers_temp.append(self.answers[cls])
                support_x.append(np.array(source_ids_temp))  
                support_x.append(np.array(source_mask_temp))
                support_x.append(np.array(target_ids_temp)) 
                support_x.append(np.array(answers_temp))
                query_x.append(np.array([self.source_ids[i]]))
                query_x.append(np.array([self.source_mask[i]]))
                query_x.append(np.array([self.target_ids[i]]))
                query_x.append(np.array([self.answers[i]]))
                               
                self.support_x_batch.append(support_x)  # append set to current sets
                self.query_x_batch.append(query_x)  # append sets to current sets
                self.query_x_index.append([i])
        elif self.mode =='test':
            train_pt = os.path.join(self.output_dir, 'train.pt')
            train_inputs = []
            with open(train_pt, 'rb') as f:
                for _ in range(4):
                    train_inputs.append(pickle.load(f)) #pickle.load() the type is array
            source_ids_train, source_mask_train, target_ids_train, answers_train = train_inputs
            for i in range(taskNum):  # for each batch/task
                k_shot_Index = relevance(i, self.mode, self.input_dir,self.k_shot)
                support_x = [] # n_way list, per list is a class list 
                query_x = []
                source_ids_temp = []
                source_mask_temp = []
                target_ids_temp = []
                answers_temp = []
                for cls in k_shot_Index:
                    source_ids_temp.append(source_ids_train[cls])
                    source_mask_temp.append(source_mask_train[cls])
                    target_ids_temp.append(target_ids_train[cls])
                    answers_temp.append(answers_train[cls])
                support_x.append(np.array(source_ids_temp))  
                support_x.append(np.array(source_mask_temp))
                support_x.append(np.array(target_ids_temp)) 
                support_x.append(np.array(answers_temp))
                query_x.append(np.array([self.source_ids[i]]))
                query_x.append(np.array([self.source_mask[i]]))
                query_x.append(np.array([self.target_ids[i]]))
                query_x.append(np.array([self.answers[i]]))
                               
                self.support_x_batch.append(support_x)  # append set to current sets
                self.query_x_batch.append(query_x)  # append sets to current sets
                self.query_x_index.append([i]) # query set index in the raw test.json 
    
    def __getitem__(self, index):
        support_x = torch.LongTensor(self.support_x_batch[index]) # append set to current sets
        query_x = torch.LongTensor(self.query_x_batch[index]) # append sets to current sets
        query_x_index = torch.LongTensor(self.query_x_index[index])   
        return support_x, query_x, query_x_index

    def __len__(self):
        return self.taskNum

