# BAIT_SFUDA

Code for ['Unsupervised Domain Adaptation without Source Data by Casting a BAIT'](https://arxiv.org/abs/2010.12427)



### Evaluation and training


First training model on the source data.

> python train_source.py

Then adapting source model to target domain, with only the unlabeled target data:

> python train_target.py

### Acknowledgement

The codes are borrowed from [SHOT (ICML 2020, also source-free)](https://github.com/tim-learn/SHOT).

