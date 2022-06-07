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

## Movie analysis

This is a pet-project for which I scraped information about top-1000 popular movies and top-1000 popular series.

* [Research on the quality of localization of movie titles](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/movie_dataset/localization_quality_analysis/movie_title_translation.ipynb) `Multilingual Sentence Embeddings` `Text Similarity`  
  The goal of this research is to find out:

  * How similar Russian titles and original titles are in general?
  * Can we split dissimilar pairs (`Russian title`::`original title`) into groups by root cause?

  [Implementation](https://github.com/Extremesarova/mini_projects/tree/main/movie_dataset/localization_quality_analysis) includes:
  * **Exploratory Data Analysis**
    * Exploring dataset
    * Cleaning dataset
      * Removing year from the `russian_title` column
      * Removing duplicates
      * Working with missing values
  * **Analysis**:
    * Computing several similarity scores for different multilingual embeddings using [Sentence-Transformers](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models):
      * distiluse-base-multilingual-cased-v1
      * distiluse-base-multilingual-cased-v2
      * paraphrase-multilingual-MiniLM-L12-v2
      * paraphrase-multilingual-mpnet-base-v2
      * LaBSE
    * Calculating single similarity score for every pair `russian_title`::`original_title`
    * Analyzing results
      * Computing descriptive statistics for similarity score vector
      * Plotting distribution plot
      * Splitting dissimilar pairs `russian_title`::`original_title` into groups with reasoning

P.S. More to come!
