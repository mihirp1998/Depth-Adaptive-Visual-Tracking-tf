# Depth Adaptive Visual Tracking
First ever open source Implementation of Computation Adaptive Siamese network for Visual Tracking
## Introduction

The following is an Unofficial implementation of [Depth-Adaptive Computational Policies for Efficient Visual Tracking](https://arxiv.org/pdf/1801.00508.pdf) by Chris Ying and Katerina Fragkiadaki. 

The folllowing topics are covered by my project:
- [x] **Data-Preprocessing**. Key and Search frame extraction from Imagenet 2017 VID dataset
- [x] **Intermediate Supervision VGG Model**. Built using Intermediate Supervision as mentioned in Paper.
- [x] **Budgeted Gating Loss**. Implemented the g* function mentioned in Paper with Shallow Feature Extractor.
- [x] **Hard Gating for Evaluation**. Hard gating which stops the computation when confidence score exceeds threshold.
- [x] **Readability**. The code is very clear,well documented and consistent.

## Model Keys




## Prerequisite
The main requirements can be installed by:
```bash
pip install -r requirements.txt
```

## Data Collection and Preprocessing

One can download the ImageNet Vid dataset from the [link](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php)

The data can be preprocessed to Key frame & Search frame using the following code

*Change the location of the dataset from main function in the file* 
```bash
python scripts/preprocess_VID_data.py
```
Finally data can be split into train and validation and pickled by the following code

```bash
python scripts/build_VID2015_imdb.py
```
The credit for the scripts to preprocess the Visual Tracking DataSet goes to [Huazhong University of Science and Technology](https://github.com/bilylee/SiamFC-TensorFlow)

## Training
It will iteratively train the Vgg Weights using Intermediate Supervison and then use the weights to 
train the Gated Weights.
This process will happen iteratively
```bash

python main.py train

```

## Evaluation
Hard Gating will  stop the computation when the confidence score exceeds the threshold
It will return the Cross Correlation Map,Flops Computed and the index of the block where computation stopped

```bash

python main.py eval

```

