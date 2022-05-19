# Salary prediction

[Solution](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/salary_prediction/salary_prediction.ipynb)  

`Regression` `DL` `NLP`  
The task is to predict salary based on the different text and categorical features.

Dataset:

* Dimensions:
  * Train: 195814 rows
  * Test: 48954 rows
* Features:
  * Categorical columns:
    * Category
    * Company
    * LocationNormalized
    * ContractType
    * ContractTime
  * Text columns:
    * Title
    * FullDescription

Implementation includes:

* **Exploratory Data Analysis**
  * Missing Values Imputation
  * Categorical Columns Encoding
  * Target transformation
* **Modeling**:
  * Base model: Custom PyTorch dataset + Custom Transforms + **Fusion model** (Title Encoder + Description Encoder + Categorical Encoder)
  * Improved model: *In progress*
* **Explaining model predictions**: *In progress*

[Original location](https://github.com/Extremesarova/yandex_nlp_course/tree/main/week02_text_classification)

## TODO

* Outliers detection
