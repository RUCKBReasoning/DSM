# DSM
The pytorch implementation of Question Generation over Knowledge Base via Modeling Diverse Subgraphs with Meta-learner.

Requirements
====
Create a virtual environment first via:
```
$ conda activate -n your_env_name python 3.8.5 pip
```
Install all the required tools using the following command:
```
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

$ pip install -r requirements.txt
```

How to run
====
(1) Create a folder to store results.
```
$ mkdir output_WQ
```

(2) To run the example, execute:
```
$ python bart_train.py --epoch 30 --input_dir dataset/WQ --output_dir './output_WQ' --update_lr 5e-5 --meta_lr 3e-5 --model_name_or_path 'facebook/bart-base'
```
