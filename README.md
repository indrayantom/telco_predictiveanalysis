# ✨ Telco Customer Churn : Predictive Analysis ✨ 
The customer base dataset used in this work is made available by IBM and downloaded from Kaggle. It is related to an anonymous telecom company and contains 7043 customers data with 21 attributes where each row represents a customer and each column contains customer’s attributes. Link for the dataset : https://www.kaggle.com/blastchar/telco-customer-churn. For your information, this project is actually a development of the last project, namely Telco Customer Churn : EDA. The documentation of this project is written in LaTeX.

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![latex](https://img.shields.io/badge/LaTeX-47A141?style=for-the-badge&logo=LaTeX&logoColor=white)

Just in case you didn't know, the telco analysis indra.html file contains both the codes and the explanation. In addition, the professional writing of the analysis is also available in a .pdf file. You also can view the docs [here](https://indrayantom.github.io/telco_custmer_dea/) . Feel free to download and clone this repo 😃😉.

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
This stacked bar plot clearly shows the customers who have churned. First, the values of tenure and monthly charges were discretized/binned into 6 quantiles to create this stacked bar plot.
![binning](https://user-images.githubusercontent.com/92590596/145628097-28917258-373b-4549-87af-6f2ba10b0161.jpg)
The result appears an instinctive result as the churn likelihood gets smaller as the membership time gets longer. It also tells that more than 50\% of customers who only subscribed less than 4 months prefer to churn (mostly even churn in their first month). From MonthlyCharges binning, it can be seen that the premium customers who are billed more than 70 dollars per month are more likely to churn compared to other customers with less bill. From the business perspective, it is surely more beneficial for the company to have a great focus improving on the premium services since those services have more lost customers and donate more month-to-month income for the company. Another interesting result is the customer who only subscribes for basic service seems quite satistied with the service quality and less likely to churn.

## References
http://www2.bain.com/Images/BB_Prescription_cutting_costs.pdf (explain why customer retention is very important for the company's life)
