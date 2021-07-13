# Logical-Reasoning-Reading-Comprehension-ReClor
Here is the code for the **#6** to the ReClor Logical Reasoning Reading Comprehension leaderboard. 

![image](https://user-images.githubusercontent.com/23516191/125377937-f4415080-e3e1-11eb-897d-48350be6792f.png)

Here is the link for the ReClor leaderboard. We are the team `qbao775`. The method we used is `RoBERTa-large` finetuned on `MNLI` dataset. In the first submission, we use the `RoBERTa-large-mnli` (https://huggingface.co/roberta-large-mnli) from the Huggingface. 

**[ReClor Leaderboard]** https://eval.ai/web/challenges/challenge-page/503/leaderboard/1347

We also finetune a RoBERTa-large-mnli by ourselves. The finetuning code is in the `finetune.py`.

The files under the MNLI can be downloaded from here, which are organized from the original MNLI website.
https://drive.google.com/drive/folders/159eVK6IsKRvzJPwdawJphnfMBr9MtDtO?usp=sharing

**[MNLI Project page]** https://www.nyu.edu/projects/bowman/multinli/

##  How to run the code?
1. Before you run the code, you need to download the MNLI folder from the Google drive link and put it under the MNLI folder.

2. Our **#6** submission code is the `run_roberta_large_mnli.sh` which located in the `scripts` folder. You can run it directly.

3. If you want to finetune a RoBERTa-large-mnli by yourself, you can try to run the `finetune.py`.
