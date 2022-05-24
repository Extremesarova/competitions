# Mini projects

## List of projects

* [Churn prediction](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/churn/churn_prediction_front.ipynb) `Binary Classification` `Classic ML` `Tabular`  
The task is to predict churn of telecom company customers.  
[Implementation](https://github.com/Extremesarova/mini_projects/tree/main/churn) includes:
  * **Exploratory Data Analysis**
    * Missing Values Imputation
    * Categorical Columns Encoding
    * Data Normalization
  * **Modeling**:
    * Baseline: SimpleImputer + StandardScaler + OneHotEncoder + **Logistic regression** + GridSearchCV
    * Final: **CatBoostClassifier** + Hyperparameter Search (Optuna)

* [Salary prediction](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/salary_prediction/salary_prediction.ipynb) `Regression` `DL` `NLP`  
The task is to predict salary based on the different text and categorical features.  
[Implementation](https://github.com/Extremesarova/mini_projects/tree/main/salary_prediction) includes:
  * **Exploratory Data Analysis**
    * Categorical Columns Encoding
    * Target transformation
  * **Modeling**:
    * Baseline: Custom PyTorch dataset + Custom Transforms + Fusion model (Title Encoder + Description Encoder + Categorical Encoder )
    * Improved model: *In progress*
  * **Explaining model predictions**: *In progress*

* [Simpsons classification](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/simpsons_classification/simpsons-classification-using-pytorch-guidelines.ipynb) `Multiclass classification` `DL` `CV` `Transfer Learning`  
The task is to build classifier using Convnets to classify images from [Simpsons](https://www.kaggle.com/competitions/journey-springfield) series onto 42 classes  
[Implementation](https://github.com/Extremesarova/mini_projects/tree/main/simpsons_classification) includes:
  * **Data preparation**:
    * Label Encoding
  * **Modeling**:
    * Baseline: Custom PyTorch dataset + Torchvision Transforms + Finetuning `vgg16_bn`
    * Improved model: *In progress*

P.S. More to come!
