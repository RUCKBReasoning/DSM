import  torch
import torch.nn as nn
import torch.optim as optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import sys
from    learner import Learner
from    copy import deepcopy
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk
import json
import collections
from copy import copy

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, flag, device):
        """
        :param args:
        """
        super(Meta, self).__init__()
        self.device = device
        self.flag = flag #flag=1 means it is forward, otherwise it is finetune
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.weight_decay = args.weight_decay
        self.adam_epsilon = args.adam_epsilon
        self.epoch = args.epoch
        self.max_grad_norm = args.max_grad_norm
        self.model_name_or_path = args.model_name_or_path
        
        model_class, tokenizer_class = (BartForConditionalGeneration, BartTokenizer)
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
       
        self.net = model_class.from_pretrained(args.model_name_or_path)  
        # load WQ pretrained model
        self.net.load_state_dict(torch.load('./param_bart_base_best.pt')) 

        self.net = torch.nn.DataParallel(self.net, device_ids = [0, 1, 2])
        self.net = self.net.to(self.device)  
                        
        self.meta_optim = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.meta_lr)
        self.inner_optim = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.update_lr)

 
   # clip_grad_by_norm_ is not used
    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm = total_norm + param_norm.item() ** 2
            counter = counter + 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter
    
    # get inner loop parameters, return inner loop parameter dict "names_weights_copy"
    def get_inner_loop_parameter_dict(self, params):
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                param_dict[name] = param.clone()
                param_dict[name].retain_grad() #clone node can compulate grad
        return param_dict
    # update inner parameters, return updated parameters dict
    def update_params(self, inner_loss, named_params, parameters, update_lr):
        self.net.zero_grad()
        grads = torch.autograd.grad(inner_loss, parameters, create_graph = True)
        updated_names_weights_dict = dict()
        for (name, param), grad in zip(named_params, grads):
            updated_names_weights_dict[name] = param - update_lr * grad
            updated_names_weights_dict[name].retain_grad()
        return updated_names_weights_dict


    def forward(self, x_spt, x_qry, query_index):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param x_qry:   [b, querysz, c_, h, w]
        :return:
        """
        self.flag = 1
        task_num = x_spt.size(0) #task_num 4
        setsz = x_spt.size(2) #setsz 5
        querysz = x_qry.size(2) ##dim=1 element size, x_qry type is tensor,querysz 75  
        corrects_1 = [0 for _ in range(self.update_step + 1)]
        corrects_2 = [0 for _ in range(self.update_step + 1)]
        corrects_3 = [0 for _ in range(self.update_step + 1)]
        corrects_4 = [0 for _ in range(self.update_step + 1)]
        corrects_r = [0 for _ in range(self.update_step + 1)]            

        # grads = [0 for _ in range(len(list(self.net.parameters())))]
        grads = [0 for i in self.net.parameters() if i.requires_grad==True]
        print("len:", len(grads))
        torch.save(self.net.module.state_dict(), './param_3.pt', _use_new_zipfile_serialization = False) #save the model the initial parameter
        for i in range(task_num):
            if i!=0:
                self.net.module.load_state_dict(torch.load('./param_3.pt'))
                
            # 1. run the i-th task and compute loss for k=0
          
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                pad_token_id = self.tokenizer.pad_token_id
                source_ids, source_mask, y = x_qry[i][0], x_qry[i][1], x_qry[i][2]
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone()
                lm_labels[y[:, 1:] == pad_token_id] = -100
                # model input
                inputs = {
                    "input_ids": source_ids,
                    "attention_mask": source_mask,
                    "decoder_input_ids": y_ids,
                    "labels": lm_labels,
                }
                bleu, _ = self.test(self.net, self.tokenizer, x_qry[i], query_index[i])               
                corrects_1[0] = corrects_1[0] + bleu[0]
                corrects_2[0] = corrects_2[0] + bleu[1]
                corrects_3[0] = corrects_3[0] + bleu[2]
                corrects_4[0] = corrects_4[0] + bleu[3]
                corrects_r[0] = corrects_r[0] + bleu[4]
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = x_spt[i][0],x_spt[i][1],x_spt[i][2]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            #model input
            inputs = {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": y_ids,
                "labels": lm_labels,
            }
                        
            outputs = self.net(**inputs)            
            loss = outputs[0]
            self.inner_optim.zero_grad()
            loss.mean().backward()
            self.inner_optim.step() 
           
            with torch.no_grad():
                bleu, _ = self.test(self.net, self.tokenizer, x_qry[i], query_index[i])  
                corrects_1[1] = corrects_1[1] + bleu[0]
                corrects_2[1] = corrects_2[1] + bleu[1]
                corrects_3[1] = corrects_3[1] + bleu[2]
                corrects_4[1] = corrects_4[1] + bleu[3]
                corrects_r[1] = corrects_r[1] + bleu[4]        

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                
                pad_token_id = self.tokenizer.pad_token_id
                source_ids, source_mask, y = x_spt[i][0], x_spt[i][1], x_spt[i][2]
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone()
                lm_labels[y[:, 1:] == pad_token_id] = -100
                # model input
                inputs = {
                    "input_ids": source_ids,
                    "attention_mask": source_mask,
                    "decoder_input_ids": y_ids,
                    "labels": lm_labels,
                }
                # self.net.named_parameters() = fast_weights
                outputs = self.net(**inputs)
                loss1 = outputs[0]
                self.inner_optim.zero_grad()
                loss1.mean().backward()
                self.inner_optim.step()
                with torch.no_grad():
                    bleu, _ = self.test(self.net, self.tokenizer, x_qry[i], query_index[i])
                    corrects_1[k+1] = corrects_1[k+1] + bleu[0]
                    corrects_2[k+1] = corrects_2[k+1] + bleu[1]
                    corrects_3[k+1] = corrects_3[k+1] + bleu[2]
                    corrects_4[k+1] = corrects_4[k+1] + bleu[3]
                    corrects_r[k+1] = corrects_r[k+1] + bleu[4]  
                    # corrects[k+1] = corrects[k+1] + bleu
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = x_qry[i][0], x_qry[i][1], x_qry[i][2]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            # model 
            inputs = {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": y_ids,
                "labels": lm_labels,
            }
            outputs = self.net(**inputs)
            loss_q = outputs[0]
            self.inner_optim.zero_grad()
            loss_q.mean().backward()
            idex = 0
            for param1 in self.net.parameters():
                if param1.requires_grad == True:
                    grads[idex] += param1.grad 
                    idex += 1
                   
        # end of all tasks
        # sum over all losses on query set across all tasks
        # optimize theta parameters    
        grads = [gr/task_num for gr in grads]
        self.meta_optim.zero_grad()
        enumerator = 0
        for par in self.net.parameters():
            if par.requires_grad==True:
                par.grad = grads[enumerator]
                enumerator += 1
       
        self.net.module.load_state_dict(torch.load('./param_3.pt'))
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm) 
        self.meta_optim.step()
        count = 0
        for xx in self.net.parameters():
            if xx.grad is not None:
                count+=1
        bleu_1 = np.array(corrects_1)/task_num   
        bleu_2 = np.array(corrects_2)/task_num  
        bleu_3 = np.array(corrects_3)/task_num
        bleu_4 = np.array(corrects_4)/task_num
        rouge_l = np.array(corrects_r)/task_num
        return bleu_1, bleu_2, bleu_3, bleu_4, rouge_l

       def finetunning(self, x_spt, x_qry, query_index, mode):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param x_qry:   [querysz, c_, h, w]
        :query_index: query_set index in the raw train/test data
        :return:
        """
        if mode == 'test':
            self.flag = 0
        
        querysz = x_qry.size(0)
        predList = []
        # corrects = [0 for _ in range(self.update_step_test + 1)]
        corrects_1 = [0 for _ in range(self.update_step_test + 1)]
        corrects_2 = [0 for _ in range(self.update_step_test + 1)]
        corrects_3 = [0 for _ in range(self.update_step_test + 1)]
        corrects_4 = [0 for _ in range(self.update_step_test + 1)]
        corrects_r = [0 for _ in range(self.update_step_test + 1)] 
        loss_list = [0 for _ in range(self.update_step_test)]
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        up_optimizer = optim.Adam(net.parameters(), lr = self.update_lr)              
       
        with torch.no_grad():
            # [setsz, nway]                       
            bleu, pred = self.test(net, self.tokenizer, x_qry[0], query_index[0])
            corrects_1[0] = corrects_1[0] + bleu[0]
            corrects_2[0] = corrects_2[0] + bleu[1]
            corrects_3[0] = corrects_3[0] + bleu[2]
            corrects_4[0] = corrects_4[0] + bleu[3]
            corrects_r[0] = corrects_r[0] + bleu[4]
            predList.append(pred)
        net.train()     
        # this is the loss and accuracy after the first update
        # with torch.no_grad():
            # net.named_parameters() = fast_weights
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = x_spt[0][0],x_spt[0][1],x_spt[0][2]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        # model
        inputs = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y_ids,
            "labels": lm_labels,
        }
        outputs = net(**inputs)
        loss = outputs[0]
        up_optimizer.zero_grad()
        loss.mean().backward()
        up_optimizer.step()
        loss_list[0] = loss_list[0] + loss.mean().item()
        with torch.no_grad():
            bleu, pred = self.test(net, self.tokenizer, x_qry[0], query_index[0])
            # corrects[1] = corrects[1] + bleu
            corrects_1[1] = corrects_1[1] + bleu[0]
            corrects_2[1] = corrects_2[1] + bleu[1]
            corrects_3[1] = corrects_3[1] + bleu[2]
            corrects_4[1] = corrects_4[1] + bleu[3]
            corrects_r[1] = corrects_r[1] + bleu[4]
            predList.append(pred)         

        for k in range(1, self.update_step_test):
            net.train()
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = x_spt[0][0], x_spt[0][1], x_spt[0][2]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            # model
            inputs = {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": y_ids,
                "labels": lm_labels,
            }
            # net.named_parameters = fast_weights
            outputs = net(**inputs)
            loss = outputs[0]            
            up_optimizer.zero_grad()
            loss.mean().backward()
            up_optimizer.step()
            loss_list[k] = loss_list[k] + loss.mean().item()
            with torch.no_grad():
                # [setsz]
                bleu, pred = self.test(net, self.tokenizer, x_qry[0], query_index[0])
                # corrects[k+1] = corrects[k+1] + bleu
                corrects_1[k+1] = corrects_1[k+1] + bleu[0]
                corrects_2[k+1] = corrects_2[k+1] + bleu[1]
                corrects_3[k+1] = corrects_3[k+1] + bleu[2]
                corrects_4[k+1] = corrects_4[k+1] + bleu[3]
                corrects_r[k+1] = corrects_r[k+1] + bleu[4] 
                predList.append(pred)
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = x_qry[0][0], x_qry[0][1], x_qry[0][2]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            # model
            inputs = {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": y_ids,
                "labels": lm_labels,
            }
           
            outputs = net(**inputs)
            loss_q = outputs[0]                         
            
        del net
        maxIndex = corrects_4.index(max(corrects_4)) # return corrects_4's index
        predMaxQ = predList[maxIndex]
        bleu_1 = np.array(corrects_1)  
        bleu_2 = np.array(corrects_2) 
        bleu_3 = np.array(corrects_3)
        bleu_4 = np.array(corrects_4)
        rouge_l = np.array(corrects_r)
        loss_list = np.array(loss_list)
        return  bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, loss_list, predMaxQ
     def test(self, model, tokenizer, data, query_index):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model = model.to(self.device)
        model = model.module.to(self.device)
        model.eval()
        with torch.no_grad():
            all_outputs = []
            source_ids, source_mask= [x.to(self.device) for x in data[:2]]
            outputs = model.generate(
                input_ids = source_ids,
                max_length = 512,
            )
            all_outputs.extend(outputs.cpu().numpy())
        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
       
        test_set = []
        if self.flag == 1:
            with open(os.path.join(self.input_dir, 'train.json')) as f:
                for line in f.readlines():
                    line = line.strip()
                    line = json.loads(line)
                    test_set.append(line)
        elif self.flag == 0:
            with open(os.path.join(self.input_dir, 'test.json')) as f:
                for line in f.readlines():
                    line = line.strip()
                    line = json.loads(line)
                    test_set.append(line)
        groundTruth = []
        for item in test_set:
            question = item['outSeq']
            groundTruth.append(question)
        ground = groundTruth[query_index[0]]
        pred = outputs[0]
        l = pred.strip().split(" ")
        ll = l[-1].split("?")
        if ll[0]!="":
            l[-1] = ll[0]
            l.append("?")
            l = " ".join(l)
            pred = l
              
        score = []
        pred = nltk.word_tokenize(pred.strip())
        ground = nltk.word_tokenize(ground.strip())
        
        pred_new = ' '.join(pred)
        ground_new = ' '.join(ground)
        if len(pred) == 0:
            pred_new = ' '
        rouge = Rouge()
        re_pred = pred_new
        rouge_score = rouge.get_scores(pred_new, ground_new)
        r_score = rouge_score[0]['rouge-l']['r'] #召回率
        r_score = round(r_score*100,4)

        
        bleu4 = sentence_bleu([ground],pred,weights=(0.25, 0.25, 0.25, 0.25)) #bleu4
        bleu4 = round(bleu4*100,4)
        bleu3 = sentence_bleu([ground],pred,weights=(0.33, 0.33, 0.33, 0))
        bleu3 = round(bleu3*100,4)
        bleu2 = sentence_bleu([ground],pred,weights=(0.5, 0.5, 0, 0))
        bleu2 = round(bleu2*100,4)
        bleu1 = sentence_bleu([ground],pred,weights=(1.0, 0, 0, 0))
        bleu1 = round(bleu1*100,4)
        # print("ground:", ground, type(ground))
        # print("pred:", pred, type(pred))
       
        score.append(bleu1)
        score.append(bleu2)
        score.append(bleu3)
        score.append(bleu4)
        score.append(r_score)
        return score, re_pred
def main():
    pass
if __name__ == '__main__':
    main()
