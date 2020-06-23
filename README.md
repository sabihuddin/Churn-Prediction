# Data-Practicum-1

## Telco Customer Churn

### Introduction

#### Telco Customer Churn is a data set taken from Kaggle website (www.kaggle.com/blastchar/telco-customer-churn).The context of the data set is to predict customer behavior.With the help of this data we will learn about some of the basic marketing analytical skills.We will create our own churn models, perform customer segmentation and make prediction using various machine learning model. 




```python
# Import the pandas library as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#import tree modulde from scikit-learn
from sklearn import tree
from sklearn.model_selection import train_test_split
#for calculating the model performance
from sklearn.metrics import accuracy_score
```


```python
# Read 'Data.csv' into a DataFrame named data
data = pd.read_csv('C:/Users/sabih/OneDrive/Desktop/Data2.csv')
```


```python
# Examine the head of the DataFrame
print(data.head())

# Examine the shape of the DataFrame
print(data.shape)
```

       customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService  \
    0  7590-VHVEG  Female              0     Yes         No       1           No   
    1  5575-GNVDE    Male              0      No         No      34          Yes   
    2  3668-QPYBK    Male              0      No         No       2          Yes   
    3  7795-CFOCW    Male              0      No         No      45           No   
    4  9237-HQITU  Female              0      No         No       2          Yes   
    
          MultipleLines InternetService OnlineSecurity  ... DeviceProtection  \
    0  No phone service             DSL             No  ...               No   
    1                No             DSL            Yes  ...              Yes   
    2                No             DSL            Yes  ...               No   
    3  No phone service             DSL            Yes  ...              Yes   
    4                No     Fiber optic             No  ...               No   
    
      TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling  \
    0          No          No              No  Month-to-month              Yes   
    1          No          No              No        One year               No   
    2          No          No              No  Month-to-month              Yes   
    3         Yes          No              No        One year               No   
    4          No          No              No  Month-to-month              Yes   
    
                   PaymentMethod MonthlyCharges  TotalCharges Churn  
    0           Electronic check          29.85         29.85    No  
    1               Mailed check          56.95        1889.5    No  
    2               Mailed check          53.85        108.15   Yes  
    3  Bank transfer (automatic)          42.30       1840.75    No  
    4           Electronic check          70.70        151.65   Yes  
    
    [5 rows x 21 columns]
    (7043, 21)
    


```python
# Count the number of missing values in each column
print(data.isnull().sum())
```

    customerID          0
    gender              0
    SeniorCitizen       0
    Partner             0
    Dependents          0
    tenure              0
    PhoneService        0
    MultipleLines       0
    InternetService     0
    OnlineSecurity      0
    OnlineBackup        0
    DeviceProtection    0
    TechSupport         0
    StreamingTV         0
    StreamingMovies     0
    Contract            0
    PaperlessBilling    0
    PaymentMethod       0
    MonthlyCharges      0
    TotalCharges        0
    Churn               0
    dtype: int64
    

There are no null values in the data set.


```python
# Print the data types of dataset
print(data.dtypes)
```

    customerID           object
    gender               object
    SeniorCitizen         int64
    Partner              object
    Dependents           object
    tenure                int64
    PhoneService         object
    MultipleLines        object
    InternetService      object
    OnlineSecurity       object
    OnlineBackup         object
    DeviceProtection     object
    TechSupport          object
    StreamingTV          object
    StreamingMovies      object
    Contract             object
    PaperlessBilling     object
    PaymentMethod        object
    MonthlyCharges      float64
    TotalCharges         object
    Churn                object
    dtype: object
    

##### Since our whole data reloves around the number of chruners and non churners hence we will convert the churn column type into integer 


```python
data.loc[data.Churn=='No','Churn'] = 0
data.loc[data.Churn=='Yes','Churn'] = 1
```

##### It looks like the data set can be divided into two categories: Categorical data: gender, payment method etc. Numerical data: tenure, monthly charges etc.


```python
# Convert 'No internet service' to 'No'
columns = ['OnlineBackup', 'StreamingMovies', 'DeviceProtection', 'TechSupport', 'OnlineSecurity', 'StreamingTV']

for i in columns:
    data[i] = data.replace({'No internet service' : 'No'})
```


```python
# Replace spaces with null values
data['TotalCharges'] = data["TotalCharges"].replace(" ",np.nan)

# Drop null values of 'Total Charges' column
data = data[data["TotalCharges"].notnull()]
data = data.reset_index()[data.columns]

# Convert 'Total Charges' column values to float data type
data["TotalCharges"] = data["TotalCharges"].astype(float)
```


```python
# Express Gender in terms of Churn
By_Gender = data.groupby('gender').Churn.mean()
print(data.groupby('gender').Churn.mean())
%matplotlib inline
By_Gender.plot(kind='bar')
```

    gender
    Female    0.269209
    Male      0.261603
    Name: Churn, dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x1f7473aff60>




![png](output_11_2.png)


##### Female customers are more likely to churn in comparison to male customers


```python
# Express PaymentMethod in terms of Churn
Payment_medium = data.groupby('PaymentMethod').Churn.mean()
print(data.groupby('PaymentMethod').Churn.mean())
%matplotlib inline
Payment_medium.plot(kind='bar')
```

    PaymentMethod
    Bank transfer (automatic)    0.167098
    Credit card (automatic)      0.152431
    Electronic check             0.452854
    Mailed check                 0.191067
    Name: Churn, dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x1f74739d1d0>




![png](output_13_2.png)


##### It seems like customer perfer automatic payment to be the best suitable medium for payment


```python
# Express Contracts in terms of Churn
By_Contract = data.groupby('Contract').Churn.mean()
print(data.groupby('Contract').Churn.mean())
%matplotlib inline
By_Contract.plot(kind='bar')
```

    Contract
    Month-to-month    0.427097
    One year          0.112695
    Two year          0.028319
    Name: Churn, dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x1f7478bff60>

