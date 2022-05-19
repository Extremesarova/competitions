# Churn prediction
[Solution](https://nbviewer.org/github/Extremesarova/mini_projects/blob/main/churn/churn_prediction_front.ipynb)  

`Binary Classification`  
The task is to predict churn of telecom company customers.  

Kaggle competition: [Предсказание оттока пользователей](https://www.kaggle.com/competitions/advanced-dls-spring-2021/submissions)  

Dataset:

* Dimensions:
  * Train: 5282 rows
  * Test: 1761 rows
* Features:
  * Categorical columns:
    * Sex
    * IsSeniorCitizen
    * HasPartner
    * HasChild
    * HasPhoneService
    * HasMultiplePhoneNumbers
    * HasInternetService
    * HasOnlineSecurityService
    * HasOnlineBackup
    * HasDeviceProtection
    * HasTechSupportAccess
    * HasOnlineTV
    * HasMovieSubscription
    * HasContractPhone
    * IsBillingPaperless
    * PaymentMethod
  * Numerical columns:
    * ClientPeriod
    * MonthlySpending
    * TotalSpent

Implementation includes:

* **Exploratory Data Analysis**
  * Missing Values Imputation
  * Categorical Columns Encoding
  * Data Normalization
* **Modeling**:
  * Baseline: SimpleImputer + StandardScaler + OneHotEncoder + **Logistic regression** + GridSearchCV
  * Final: **CatBoost** + Hyperparameter Search (Optuna)

[Original location](https://github.com/Extremesarova/deep_learning_school/tree/main/part_1/3_kaggle)

## TODO

* Outliers detection
