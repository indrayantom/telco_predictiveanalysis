# ✨ Telco Customer Churn : Predictive Analysis ✨ 
The customer base dataset used in this work is made available by IBM and downloaded from Kaggle. It is related to an anonymous telecom company and contains 7043 customers data with 21 attributes where each row represents a customer and each column contains customer’s attributes. Link for the dataset : https://www.kaggle.com/blastchar/telco-customer-churn. For your information, this project is actually a development of the last project, namely Telco Customer Churn : EDA. The documentation of this project is written in LaTeX.

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![latex](https://img.shields.io/badge/LaTeX-47A141?style=for-the-badge&logo=LaTeX&logoColor=white)

Just in case you didn't know, the telco analysis indra.html file contains both the codes and the explanation. In addition, the professional writing of the analysis is also available in a .pdf file. You also can view the Google Collab docs [here](https://colab.research.google.com/drive/1_DIwM4A7kMZOEInNVh2GWwKaJ5IhoBFT#scrollTo=bNXeBG2UykdD) . Feel free to download and clone this repo 😃😉.

## Objectives 
This work is carried out to answer this problem :

- Which machine learning model is the most accurate on predicting the churned customer and how is the performance? (Random Forest, KNN, and Logistic Regression are the only models that will be examined for now)

## Libraries
Libraries such as [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/) are the most commonly used in the analysis. However, I also used [sklearn](https://scikit-learn.org/stable/) to conduct the predictive analysis with some classification models.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score, classification_report,f1_score,precision_recall_curve,roc_curve
from imblearn import over_sampling
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
```
Also, thanks to Jefferson Silveira and Alastir McLean for providing a free LaTeX template for this project. Download or copy the LaTeX code through Overleaf [here](https://www.overleaf.com/project/61a734eae015a6257592d565) .
## Result Preview
By Hyperparameter Tuning and Decision threshold adjusting, it is found that Logistic Regression gives the best result compared to KNN and Random Forest in terms of AUC and F1 score.
![Fi](https://user-images.githubusercontent.com/92590596/156796504-440be765-e057-48a9-b559-ca07f7849550.jpg)

After being tuned with GridSearchCV method and adjusted to 0.3 decision/probability threshold, the improvement becomes significant compared to the default model as the Recall and F1 score are increased to 76\% (**+23\%**) and 63\% (**+5\%**) respectively . With an AUC score of 83\% (**+0\%**), those metrics are successfully increased without a lot of reduction in accuracy, only **3\%** lower from the default model with the score of 77\%.

Futhermore, the feature importance (coefficient) of the Logistic regression model can be seen on above figure. Notice that the coefficients are both positive and negative. It can be elaborated as the predictor of Class 1 (Churn Yes) has positive coefficient whereas the predictor of Class 0 (Churn No) has negative coefficient. Overall, it is evident that the graph  is already in accordance to the result of EDA project carried out before on similar dataset . Contract, tenure, InternetService, PaymentMethod and some additional internet services such as TechSupport and Streaming are considered as the key features on which the business strategists should focus to improve the level of satisfaction and retent the customer. Read the EDA [here](https://github.com/indrayantom/telco_custmer_dea).

## References
http://www2.bain.com/Images/BB_Prescription_cutting_costs.pdf (explain why customer retention is very important for the company's life)
