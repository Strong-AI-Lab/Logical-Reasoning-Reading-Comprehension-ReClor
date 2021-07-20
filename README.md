# Logical-Reasoning-Reading-Comprehension-ReClor
Here is the code for the **#6** to the ReClor Logical Reasoning Reading Comprehension Leaderboard. 

![image](https://user-images.githubusercontent.com/23516191/125377937-f4415080-e3e1-11eb-897d-48350be6792f.png)

Here is the link for the ReClor leaderboard. We are the team `qbao775`. The method we used is `RoBERTa-large` finetuned on `MNLI` dataset. In the first submission, we use the [RoBERTa-large-mnli] (https://huggingface.co/roberta-large-mnli) from the Huggingface. 

**[ReClor Leaderboard](https://eval.ai/web/challenges/challenge-page/503/leaderboard/1347)**

We also finetune a RoBERTa-large-mnli by ourselves. The finetuning code is in the `finetune.py`.

The files under the MNLI can be downloaded from here, which are organized from the original MNLI website.
https://drive.google.com/drive/folders/159eVK6IsKRvzJPwdawJphnfMBr9MtDtO?usp=sharing

**[MNLI Project page](https://www.nyu.edu/projects/bowman/multinli/)**

##  How to run the code?
### Environment setup
- [Python3.5+]
- [PyTorch 1.0+]
- [Transformers 2.3.0]
- [apex](https://github.com/NVIDIA/apex) - install Nvidia apex for mixed precision training
Install Python3.5+, PyTorch 1.0+, Transformers and apex

### Load existing RoBERTa-large-mnli from Huggingface
Our **#6** submission code is the `run_roberta_large_mnli.sh` which located in the `scripts` folder. You can run it directly.
1. Run the scripts in the main directory by such as `sh scripts/run_roberta_large.sh`
2. You will find `test_preds.npy` which is the test prediction result. You need to submit it to the [ReClor leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/503/leaderboard/1347).

### Finetune a RoBERTa-large-mnli by yourself
1. Before you run the code, you need to download the MNLI folder from the Google drive link and put it under the MNLI folder and `pip install -r requirements.txt` to install all needed packages.

2. Then you can run the `finetune.py`.

## Built With

 - [Torch](https://pytorch.org/) - library used to train and run models
 - [Transformers](https://huggingface.co/transformers/) - Huggingface library used to implement models
 - [Sklearn](https://scikit-learn.org/stable/) - library used to implement and evaluate models
 - [Matplotlib](https://matplotlib.org/) - main plotting library
 - [Seaborn](https://seaborn.pydata.org/) - helper plotting library for some charts
 - [NumPy](http://www.numpy.org/) - main numerical library for data vectorisation
 - [Pandas](https://pandas.pydata.org/) - helper data manipulation library
 - [Jsonlines](https://pypi.org/project/jsonlines/) - helper jsonl data manipulation library
 - [Apex](https://github.com/NVIDIA/apex/) - install Nvidia apex for mixed precision training

## Acknowledgement
Thanks for the benchmark source code provided from the ReClor group.
https://github.com/yuweihao/reclor
