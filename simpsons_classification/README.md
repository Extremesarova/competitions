# Simpsons classification

[Solution](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/simpsons_classification/simpsons-classification-using-pytorch-guidelines.ipynb)  

`Multiclass classification` `DL` `CV` `Transfer Learning`  
The task is to build classifier using Convnets to classify images from [Simpsons](https://www.kaggle.com/competitions/journey-springfield) series onto 42 classes.

Dataset:

* Dimensions:
  * Train: 20933 images
  * Test: 991 images

Implementation includes:

* **Data preparation**:
  * Label Encoding
* **Modeling**:
  * Baseline: Custom PyTorch dataset + Torchvision Transforms + Finetuning `vgg16_bn`
  * Improved model: *In progress*

[Original location](https://www.kaggle.com/code/extremesarova/simpsons-classification-using-pytorch-guidelines)

## TODO

* Improved model:
  * Error analysis
  * Fixing overfitting
  * Trying different architectures/transformations
  * Plotting loss for train and validation
  * Fixing class imbalance
