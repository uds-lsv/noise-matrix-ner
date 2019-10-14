# Feature-Dependent Confusion Matrices for Low-Resource NER Labeling with Noisy Labels

This is the repository for the publication

> Lukas Lange, Michael A. Hedderich and Dietrich Klakow
>
> **Feature-Dependent Confusion Matrices for Low-Resource NER Labeling with Noisy Labels**
>
> EMNLP 2019

## Noise-Handling Architectures
In this repository, you can find implementations for the following noisy-label neural network architectures:

* **Global Noise Matrix** from
Hedderich & Klakow: Training a Neural Network in a Low-Resource Setting on Automatically, 2018
* **Feature-Dependent Noise Matrix** from
Lange et al.: Feature-Dependent Confusion Matrices for Low-Resource NER Labeling with Noisy Labels, 2019
* **Cleaning Model** from
Veit & al.:  Learning  from  Noisy  Large-Scale  Datasets  with  Minimal Supervision, 2017
(adapted to the NLP setting)
* **Dynamic Transition Matrix** from
Luo et al.: Learning with Noise:  Enhance Distantly Supervised Relation Extraction with Dynamic Transition Matrix, 2017
(adapted to the NLP setting)

The code is written for the Named Entity Recognition (NER) setting from the paper but should be easily adaptable to other supervised learning tasks by replacing the data loader and (if needed) the base model architecture.

## Installation & Structure

```sh
conda create --name noise-matrix-ner python=3.6
source activate noise-matrix-ner
pip install numpy==1.16.2 scikit-learn==0.20.3 jupyter==1.0.0 tensorflow==1.12.0 Keras==2.2.4 
# depending on your hardware, you might want to replace tensorflow with tensorflow-gpu
# fastText needs to be installed manually from https://fasttext.cc/docs/en/support.html (do not use the version from the online pip repo)
```

The repository has the following structure

* *code*: 
  * *ner.ipynb*: Jupyter notebook with the whole experimental pipeline.
  * *layers.py*: Implementation of different special layers in Keras for the noise-handling architectures.
  * *ner_datacode.py*: Utility code for NER, data loading, word embeddings and evaluation.
  * *noisematrix.py*: Utility code for noise matrices, including visualization.
  * *experimentalsettings.py*: Utility code to store experimental configurations.
* *config*: Example configurations for the different experiments.
* *data*: Due to legal reasons, this repository only contains some dummy data. The CoNLL02/03 datasets are widely available. The Estonian NER dataset can be obtained [here](https://metashare.ut.ee/repository/browse/estonian-ner-corpus/88d030c0acde11e2a6e4005056b40024f1def472ed254e77a8952e1003d9f81e/).
 
 
## License & Citation

The code is licensed under Apache 2, so feel free to use it in your projects. If you do, please cite us as 

```
@InProceedings{Lange2019FeatureDependent,
  author = "Lange, Lukas and Hedderich, Michael A. and Klakow, Dietrich",
  title = "Feature-Dependent Confusion Matrices for Low-Resource NER Labeling with Noisy Labels",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  year = "2019",
  publisher = "Association for Computational Linguistics"
}
```

If you use the implementation of the Global Noise Matrix or the Cleaning Model, please also cite

```
@InProceedings{W18-3402,
  author = "Hedderich, Michael A. and Klakow, Dietrich",
  title = "Training a Neural Network in a Low-Resource Setting on Automatically Annotated Noisy Data",
  booktitle = "Proceedings of the Workshop on Deep Learning Approaches for Low-Resource NLP",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  pages = "12--18",
  location = "Melbourne",
  url = "http://aclweb.org/anthology/W18-3402"
}
```

## Contact
If you have any questions, feel free to contact the authors at {llange,mhedderich} at lsv. uni-saarland .de




