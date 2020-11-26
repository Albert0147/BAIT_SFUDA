# BAIT_SFUDA

Code (pytorch) for ['Unsupervised Domain Adaptation without Source Data by Casting a BAIT'](https://arxiv.org/abs/2010.12427) on VisDA.


### Training and evaluation
You need to download the [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) dataset.

1. First training model on the source data.

> python train_source.py

2. Then adapting source model to target domain, with only the unlabeled target data:

> python train_target.py


### Results in paper
![VisDA](/img/visda.png)


### Acknowledgement

The codes are based on [SHOT (ICML 2020, also source-free)](https://github.com/tim-learn/SHOT).

