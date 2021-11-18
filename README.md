# Logical-Reasoning-Reading-Comprehension-ReClor
Here is the code for the **#5** to the ReClor Logical Reasoning Reading Comprehension Leaderboard (**2021/07/28**). 

![image](https://user-images.githubusercontent.com/23516191/127267725-54938a0c-c3d9-41ae-a9a5-77095b11e523.png)

Here is the code for the **#6** to the ReClor Logical Reasoning Reading Comprehension Leaderboard (**2021/07/27**). 

![image](https://user-images.githubusercontent.com/23516191/125377937-f4415080-e3e1-11eb-897d-48350be6792f.png)

Here is the link for the ReClor leaderboard. We are the team `qbao775`. The method we used is `RoBERTa-large` finetuned on `MNLI` dataset. In the first submission, we use the [RoBERTa-large-mnli](https://huggingface.co/roberta-large-mnli) from the Huggingface. 

**[ReClor Leaderboard](https://eval.ai/web/challenges/challenge-page/503/leaderboard/1347)**

We also finetune a RoBERTa-large-mnli by ourselves. The finetuning code and hyperparameters are referred from the Huggingface (https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification).

The files under the MNLI can be downloaded from [here](https://drive.google.com/drive/folders/159eVK6IsKRvzJPwdawJphnfMBr9MtDtO?usp=sharing), which are organized from the original MNLI website.

**[MNLI Project page](https://www.nyu.edu/projects/bowman/multinli/)**

##  How to run the code?
### Environment setup
- Python3.5+
- PyTorch 1.0+
- Transformers 2.3.0
- [apex](https://github.com/NVIDIA/apex) - install Nvidia apex for mixed precision training
Install Python3.5+, PyTorch 1.0+, Transformers and apex

### Load existing RoBERTa-large-mnli from Huggingface
Our **#5** submission code (2021/07/28) is the `run_roberta_large_MNLI_PARARULE_Plus_reclor.sh` which located in the `scripts` folder. You can run it directly.
1. Before you run the scripts in the main directory by such as `run_roberta_large_MNLI_PARARULE_Plus_reclor.sh`, please run the `run_roberta_large_MNLI_PARARULE_Plus.sh` firstly and then use the lastest output model as the initialization model for the `run_roberta_large_MNLI_PARARULE_Plus_reclor.sh`.
2. Run the scripts in the main directory by such as `sh scripts/run_roberta_large_MNLI_PARARULE_Plus_reclor.sh`
3. You will find `test_preds.npy` which is the test prediction result. You need to submit it to the [ReClor leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/503/leaderboard/1347).

Our **#6** submission code (before 2021/07/28) is the `run_roberta_large_mnli.sh` which located in the `scripts` folder. You can run it directly.
1. Run the scripts in the main directory by such as `sh scripts/run_roberta_large.sh`
2. You will find `test_preds.npy` which is the test prediction result. You need to submit it to the [ReClor leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/503/leaderboard/1347).

The test predication results `test_preds.npy` submitted to the leaderboard and models can be found from [here](https://drive.google.com/drive/folders/1krlBEyBMsHGKa8i-HVCMR1l3RT4c4Ne_?usp=sharing).

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

PARARULE Plus: A Larger Deep Multi-Step Reasoning Dataset over Natural Language
https://github.com/Strong-AI-Lab/PARARULE-Plus
