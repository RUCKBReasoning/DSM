import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re
import sys
from transformers import BartTokenizer
from transformers import T5Tokenizer

def get_subkg_seq(subkg): 
    seq = []
    maskList = []
    g_nodes = subkg['g_node_names']
    g_edges = subkg['g_edge_types']
    g_adj = subkg['g_adj']
    all_subjects = []
    all_objects = []
    for key, value in g_adj.items():
        subject = g_nodes[key]
        all_subjects.append(subject)
        #relation = value.values()
        for k, relation in value.items():
            obj = g_nodes[k]
            all_objects.append(obj)            
            #WQ relation is a str
            relation = relation.strip().split('/')[-1]
            
            #PQ relation is a list 
            #relation = relation[0]
            #if relation.find('/') >= 0:
            #   relation = relation.strip().split('/')[-1]   
            if relation.find('_')!=-1:
                relation = relation.split('_')
                relation = ' '.join(relation).strip()
            fact = "{} {} {}".format(subject, relation, obj)
            seq.append(fact)
    subkg = ' </s> '.join(seq)
    # maskList = list(set(all_subjects).intersection(set(all_objects)))
    return subkg, seq, maskList 

def encode_dataset(dataset, tokenizer, test=False):
    max_seq_length = 1024
    questions = []
    answers = []
    subkgs = []
    for item in tqdm(dataset):
        question = item['outSeq']
        questions.append(question)
        subkg = item['inGraph']
        subkg, _, _ = get_subkg_seq(subkg)
        subkgs.append(subkg)
        answer = item['answers']       
        if len(answer)>1:
            answer = [', '.join(answer)]
        if len(answer)==0:
            answer = ['']
        answers = answers + answer
    s = [i +' </s> ' + j for i, j in zip(subkgs, answers)]    
    input_ids = tokenizer.batch_encode_plus(s, max_length = max_seq_length, pad_to_max_length = True, truncation = True)#与上面的tokenizer()是一样的
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
    if not test:
        target_ids = tokenizer.batch_encode_plus(questions, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
        target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    else:
        target_ids = np.array([], dtype = np.int32)
        
    answers = tokenizer(answers, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
    answers = np.array(answers['input_ids'], dtype = np.int32)
    return source_ids, source_mask, target_ids, answers 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()
    print('Loading!!!!')
 
    test_set = []
    train_set = []
    val_set = []
    with open(os.path.join(args.input_dir, 'test.json')) as f:
        for line in f.readlines():
            line = line.strip()
            line = json.loads(line)
            test_set.append(line)
    with open(os.path.join(args.input_dir, 'train.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            train_set.append(line)
    with open(os.path.join(args.input_dir, 'dev.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            val_set.append(line)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    # BART    
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(dataset,tokenizer, False)
        print(type(outputs))
        print('shape of input_ids of questions, attention_mask of questions, input_ids of sparqls, choices and answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)
if __name__ == '__main__':
    main()
