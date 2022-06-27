# DSM
The pytorch implementation of Question Generation over Knowledge Base via Modeling Diverse Subgraphs with Meta-learner.

## Requirements
1. Environments
* Create a virtual environment first via:
```
$ conda activate -n your_env_name python 3.8.5 pip
```
* Install all the required tools using the following command:
```
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

$ pip install -r requirements.txt
```
2. Dataset
* WQ : `dataset/` contains the files for WQ dataset. 
* PQ : `dataset/` contains the files for PQ dataset. 

## How to run 
1. Prepare dataset for training retriever.
```
$ python retriever/preprocess_retriever.py
```
2. Train the retriever to create learning tasks.
* Run GCL-based retriever: 
``
 $ python retriever/main_gcl.py
``
* Run DGI-based retriever: 
``
 $ python retriever/main_dgi.py
``
* Run RGCN-based retriever: 
``
 $ python retriever/main_rgcn.py
``
* Run GED retriever: 
``
$ python retriever/main_ged.py
``
* Run All-RP retriever: 
``
 $ python retriever/relation_path.py
``
3. Prepare dataset for DSM.
```
$ python preprocess.py -input_dir dataset/WQ --output_dir './output_WQ' --model_name_or_path 'facebook/bart-base'
```
4. To run the DSM, execute (Note: we take the GCL-based retriever as an example, and other retrievers are similar.):
```
$ python bart_train.py --epoch 30 --input_dir dataset/WQ --output_dir './output_WQ' --update_lr 5e-5 --meta_lr 3e-5 --model_name_or_path 'facebook/bart-base'
```
## QA performance of GRAFT-Net and NSM
We evaluate two classical KBQA models named [GRAFT-Net](https://arxiv.org/abs/1809.00782) and [NSM](https://arxiv.org/abs/2101.03737) on [WebQSP](https://aclanthology.org/P16-2033.pdf). To evaluate the quality of the generated questions by DSM, we replace part of the (question, answer) pairs in WebQSP with the generated questions. The code for GRAFT-Net and NSM can be downloaded from https://pan.baidu.com/s/1RGOZW23FbfcpkShvJkGRUw?pwd=ddsm.

