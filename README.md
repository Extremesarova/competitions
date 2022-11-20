# Mini projects

## Churn prediction

`Binary Classification` `Classic ML` `Tabular`  
[Notebook](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/churn/churn_prediction.ipynb)  
The task is to predict churn of telecom company customers.  
[Implementation](https://github.com/Extremesarova/mini_projects/tree/main/churn) includes:

* **Exploratory Data Analysis**
  * Missing Values Imputation
  * Categorical Columns Encoding
  * Data Normalization
* **Modeling**:
  * Baseline: SimpleImputer + StandardScaler + OneHotEncoder + Logistic regression + GridSearchCV
  * Final: CatBoostClassifier + Hyperparameter Search (Optuna)

## Salary prediction

`Regression` `DL` `NLP`  
[Notebook](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/salary_prediction/salary_prediction.ipynb)  
The task is to predict salary based on the different text and categorical features.  
[Implementation](https://github.com/Extremesarova/mini_projects/tree/main/salary_prediction) includes:

* **Exploratory Data Analysis**
  * Categorical Columns Encoding
  * Target transformation
* **Modeling**:
  * Baseline: Custom PyTorch dataset + Custom Transforms + Fusion model (Title Encoder + Description Encoder + Categorical Encoder)
  * Improved model: *In progress*
* **Explaining model predictions**: *In progress*

## Simpsons classification

`Multiclass classification` `DL` `CV` `Transfer Learning`  
[Notebook](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/simpsons_classification/simpsons-classification-using-pytorch-guidelines.ipynb)  
The task is to build classifier using ConvNets to classify images from [Simpsons](https://www.kaggle.com/competitions/journey-springfield) series onto 42 classes.  
[Implementation](https://github.com/Extremesarova/mini_projects/tree/main/simpsons_classification) includes:

* **Data preparation**:
  * Label Encoding
* **Modeling**:
  * Baseline: Custom PyTorch dataset + Torchvision Transforms + Finetuning `vgg16_bn`
  * Improved model: *In progress*

## Matching subscribers

`Multiclass classification` `Classic ML` `Tabular` `Feature Engineering`  
[Notebook](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/identifiers_matching/code/02_matching.ipynb)  
The task is to match customers of telecom company based on their characteristics.  
[Implementation](https://github.com/Extremesarova/mini_projects/tree/main/identifiers_matching) includes:

* **Exploratory Data Analysis**
  * Finding & fixing errors in features
  * [Memory optimization](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/identifiers_matching/code/01_optimizing_storage.ipynb)
  * Checking data integrity
* **Modeling**:
  * Baseline: TF-IDF + LogisticRegression + GridSearch
  * Final: *In progress*
* **Explaining model predictions**: ELI5
