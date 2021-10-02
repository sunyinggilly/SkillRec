# Code Description

## CONFIG.py
Configurations, "HOME_PATH" should be properly set before runing the code.

## Sampler.py
Code for sampling actions with different strategies.

## prepare_code
Preprocessing codes, such as generating the skill graph and itemsets.

## Environment
The code for the environment, including source code written in C++ and python packages for linux built with swig.

## Model
Containing our model and the baseline methods.

## Trainers
Load the training data, train the models, save the models and evaluate the models.

# Citation
If you use our models, please cite the following paper:

```
@inproceedings{sun2021cost,
  title={Cost-Effective and Interpretable Job Skill Recommendation with Deep Reinforcement Learning},
  author={Sun, Ying and Zhuang, Fuzhen and Zhu, Hengshu and He, Qing and Xiong, Hui},
  booktitle={Proceedings of the Web Conference 2021},
  pages={3827--3838},
  year={2021}
}
```
