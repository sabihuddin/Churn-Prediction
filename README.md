## Data-Practicum-1

### Telco Customer Churn

#### Introduction

Telco Customer Churn is a data set taken from Kaggle website (www.kaggle.com/blastchar/telco-customer-churn).The context of the data set is to predict customer behavior.With the help of this data we will learn about some of the basic marketing analytical skills.We will create our own churn models, perform customer segmentation and make prediction using various machine learning model. 

#### Project Description

In this project we will be performing tasks that are normally performed in while conducting marketing analytical research. Marketing concepts such as customer segmentation, churn modeling, customer life time value will be focal point of this project as we will be generating customer life time value, customer purchase pattern.We will fit logistic regression and decision tree models for churn prediction.Furthermore, we will use hyper parameter tuning to improve the model performance.

Following is the break down for weekly activity that will be performed accordingly:   
 
Week 1	Performing exploratory data analysis i.e cleaning and summarizing the data set and uploading the necessary libraries.Exploring the data by using various visualization tools. 
Week 2	Preparing the data for machine learning models by separating the numerical and categorical columns. 
Week 3	Predicting churn with logistic regression model, fitting decision tree.
Week 4	Building up more models (i.e K-Nearest Neighbor Model & Random Forest Classification Model) and calculating the accuracy of all models. 
Week 5	Using different visualization tools to visualize the findings and key factors of the data set
Week 6	Creating Confusion and other model matrix
Week 7	Reflecting on some of the models that can help in improvement of model performance 
Week 8	Preparing video and a PowerPoint presentation that will delineate the key findings of the project

For successful execution of the project different kernels and online sources will be used for getting befitting results.The objective of this project of project is to understand the core concept of marketing analytical research and how with the help of Python and different machine learning models we can anticipate the future outcome and understand the customer behavior in the most befitting manner.


First we will import all the necessary files that are required for thsi project

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

Next we will upload the data set that has been downloaded from kaggle

```python
# Read 'Data.csv' into a DataFrame named data
data = pd.read_csv('C:/Users/sabih/OneDrive/Desktop/Data2.csv')
```

We will run the necessary commandas to get a better view of the data set

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
    

Since our whole data reloves around the number of chruners and non churners hence we will convert the churn column type into integer 


```python
data.loc[data.Churn=='No','Churn'] = 0
data.loc[data.Churn=='Yes','Churn'] = 1
```

It looks like the data set can be divided into two categories: Categorical data: gender, payment method etc. Numerical data: tenure, monthly charges etc.


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


Female customers are more likely to churn in comparison to male customers


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

It seems like customer perfer automatic payment to be the best suitable medium for payment


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

Month to month contract has higher churn rate.This could indicate that company is unable to retain new customers and customer who have been using the comapnies services over one year are still using the wervices 


```python
# Express the counts as proportions
By_Tenure = data.groupby('tenure').Churn.mean()

%matplotlib inline
By_Tenure.plot(style='.')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f7474f6d30>




![png](output_17_1.png)


Same thing can be seen if we visualize the tenure against the churn rate we see that higher tenure will have lower churn rate 

#### Machine Learning models

Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.There are two types of supervised learning classificationa and regression.

Since most of the data in the data set is object in terms of data type we will have transform it in order to performe the machine learning operations in the most befitting manner


```python
#Perform One Hot Encoding using get_dummies method
data = pd.get_dummies(data, columns = ['Contract','Dependents','DeviceProtection','gender','InternetService','MultipleLines','OnlineBackup','OnlineSecurity','StreamingMovies','StreamingTV','TechSupport','PaperlessBilling','Partner','PaymentMethod','PhoneService','SeniorCitizen',], drop_first=True)
```


```python
from sklearn.preprocessing import StandardScaler

#Perform Scaling Feature
standardScaler = StandardScaler()
columns_for_ft_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges']

#Apply the feature scaling operation on dataset using fit_transform() method
data[columns_for_ft_scaling] = standardScaler.fit_transform(data[columns_for_ft_scaling])
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

```python
# For Training and Testing the dataset
X = data.drop(['Churn', 'customerID'], axis = 1)
Y = data['Churn']
Y=Y.astype('int')

```

#### Supervised machine leanining

1-Split the data into traning and testing 2-Executing the model 3-Fit the model on the training data 4-Predict valueson the testing data


```python
# Split X and Y into training and testing datasetst
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Checking the percenatge value of training dataset 
print(X_train.shape[0] / X.shape[0])
```

    0.75
    


```python
# Initiating the Logistic Regression model 
logmodel = LogisticRegression(random_state=50)

# Fit logistic regression on training data
logmodel.fit(X_train, Y_train)

# Predict churn labels on testing data
pred_test_Y = logmodel.predict(X_test)

# Measure model performance on testing data

accuracy_score(Y_test, pred_test_Y)
```

    C:\Users\sabih\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




    0.800910125142207




```python
# Initiating the model with max_depth set at 6
mytree = tree.DecisionTreeClassifier(max_depth = 6)

# Fit the model on the training data
treemodel = mytree.fit(X_train,Y_train)

# Predict values on the testing data
treemodel_pred = treemodel.predict(X_test)

# Measure model performance on testing data
accuracy_score(Y_test, treemodel_pred)
```




    0.7844141069397043



#### Other models


```python
# Initiating the K-Nearest Neighbor Model
knnmodel = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

#Fit the K-Nearest Neighbor Model
knnmodel.fit(X_train, Y_train) 
  
# Predict values on the testing data
knn_pred = knnmodel.predict(X_test)

# Measure model performance on testing data
accuracy_score(Y_test, knn_pred)
```




    0.7764505119453925




```python
# Initiating the Random Forest Classification Model
from sklearn.ensemble import RandomForestClassifier

#Fit the Random Forest Classification Model
rfmodel = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
rfmodel.fit(X_train, Y_train) 
  
#Predict the value for new, unseen data
rf_pred = rfmodel.predict(X_test)

# Measure model performance on testing data
accuracy_score(Y_test, rf_pred)
```




    0.785551763367463




```python
# Comparing model accuracy level

print("Logistic Regression;", round(accuracy_score(Y_test, pred_test_Y)*100, 2), "%")
print("Decision Tree;", round(accuracy_score(Y_test, treemodel_pred)*100, 2), "%")
print("K-Nearest Neighbor Model;", round(accuracy_score(Y_test, knn_pred)*100, 2), "%")
print("Random Forest Classification Model;", round(accuracy_score(Y_test, rf_pred)*100, 2), "%")
```

    Logistic Regression; 80.09 %
    Decision Tree; 78.44 %
    K-Nearest Neighbor Model; 77.65 %
    Random Forest Classification Model; 78.56 %
    

#### Confusion Matrix


```python
# Confusion Matrix
# Since Logestic regression model had the highest accuracy rate hence we will access its accurcay using confusion martix
cm = confusion_matrix(Y_test, pred_test_Y)
cm
```




    array([[1163,  125],
           [ 225,  245]], dtype=int64)




```python
# Predicting the Churn possiblity of each customer
data['Churn_Prob'] = logmodel.predict_proba(data[X_test.columns])[:,1]
```


```python
# Create a Dataframe showcasing probability of Churn of each customer
data[['customerID','Churn_Prob']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
```python

```

