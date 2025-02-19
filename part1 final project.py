#!/usr/bin/env python
# coding: utf-8

# Predicting Hazardous NEOs (Nearest Earth Objects)

# https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024/data

# 1-Data Importing and Cleaning

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve


# In[2]:


df = pd.read_excel("C:/Users/Lenovo/Desktop/archive1/nearest-earth-objects_1910-2024_.xlsx")
df.head()


# In[3]:


print(df.head())                                        #get overview data


# In[4]:


print(df.info())                                        #get info data


# In[5]:


print(df.describe())                                    #summary for data


# In[6]:


print(df.isnull().sum())                                #missing value


# In[7]:


print(df.dtypes)


# In[8]:


df_numeric = df.select_dtypes(include=[float, int])                         #Convert data to digital data
print(df_numeric.head())


# In[9]:


df.duplicated().sum()                         #check duplicated data


# EDA

# In[10]:


plt.figure(figsize=(10, 6))                           #Key Distributions
sns.histplot(df['absolute_magnitude'], kde=True, bins=30)
plt.title('Distribution of Absolute Magnitude')
plt.show()


# In[11]:


plt.figure(figsize=(10, 8))                                                    #Correlation Heatmap  
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)  
plt.title('Correlation Matrix')     
plt.show()


# In[12]:


fig = px.scatter(df, x='relative_velocity', y='miss_distance', color='is_hazardous')
fig.show()


# Data Preprocessing

#Handle missing values

# In[13]:


print(df.columns)


# In[14]:


df['relative_velocity'].fillna(df['relative_velocity'].mean())


# In[15]:


df['miss_distance'].fillna(df['miss_distance'].mean())


# In[16]:


df['is_hazardous'].fillna(df['is_hazardous'].mean())


#fillning missing value

# In[17]:


df['estimated_diameter_min'].fillna(df['estimated_diameter_min'].median())


# In[18]:


df['estimated_diameter_max'].fillna(df['estimated_diameter_max'].median())


#Convert necessary columns to numeric

# In[19]:


pd.to_numeric(df['absolute_magnitude'], errors='coerce')                  


#Remove Unnecessary Columns:

# In[20]:


df.drop([ 'name', 'orbiting_body'], axis=1)                           


# In[21]:


print(df['is_hazardous'].value_counts())                  # Handle imbalanced classes using SMOTE


# In[22]:


df['is_hazardous']


# In[23]:


df.drop(columns=['is_hazardous'])


# Features

# In[24]:


le = LabelEncoder()                                      # Encode categorical variables

le.fit_transform(df['orbiting_body'])


# In[25]:


scaler = StandardScaler()
scaler.fit_transform(df[['absolute_magnitude', 'relative_velocity', 'miss_distance', 'is_hazardous']])        #scaler of data


# In[26]:


features = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']
X = df[features]                                             #Feature Selection


# In[40]:


le.fit_transform(df['orbiting_body'])
X.loc[:, 'orbiting_body'] = df['orbiting_body_encoded']


# In[41]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[42]:


imputer = SimpleImputer(strategy='mean')  
X_imputed = imputer.fit_transform(X)


# In[80]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# # Model Selection

# In[75]:


X = df.drop('is_hazardous', axis=1)
y = df['is_hazardous']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[76]:


pd.cut(df['is_hazardous'], bins=[0, 5, 10, 15], labels=['Low', 'Medium', 'High'])
df['is_hazardous'].unique()


# In[58]:


df['column_encoded'] = LabelEncoder().fit_transform(df['is_hazardous'])
df['column_encoded'] = LabelEncoder().fit_transform(df['relative_velocity'])
df['column_encoded'] = LabelEncoder().fit_transform(df['miss_distance'])


# # Model Training and Evaluation

# In[66]:


X = df.drop(columns=['name', 'neo_id']) 


# In[67]:


labelencoder = LabelEncoder()
df['name'] = labelencoder.fit_transform(df['name'])


# In[68]:


X = df.drop(['is_hazardous'], axis=1)
y = df['is_hazardous']


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# In[90]:


Train a random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# In[87]:


print(f'{model_name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}'


# In[86]:


Train a logistic regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)


# In[83]:


Evaluate the models
y_pred_rf = rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test


# In[91]:


print('Random Forest Model:')
print('Precision:', precision_score(y_test, y_pred_rf))
print('Recall:', recall_score(y_test, y_pred_rf))
print('F1-Score:', f1_score(y_test, y_pred_rf))
print('AUC-ROC Curve:', roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

print('Logistic Regression Model:')
print('Precision:', precision_score(y_test, y_pred_lr))
print('Recall:', recall_score(y_test, y_pred_lr))
print('F1-Score:', f1_score(y_test, y_pred_lr))
print('AUC-ROC Curve:', roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))

















