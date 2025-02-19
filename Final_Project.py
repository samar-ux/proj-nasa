#!/usr/bin/env python
# coding: utf-8

# # Predicting Hazardous NEOs (Nearest Earth Objects)

# https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024/data

# # 1-Data Importing and Cleaning

# In[3]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report




df = pd.read_excel("C:/Users/Lenovo/Desktop/archive1/nearest-earth-objects_1910-2024_.xlsx")
df.head()




df.info()




df.describe()                                     #summary for data




df.duplicated().sum()                         #check duplicated data




print(df.isnull().sum())                                #missing value




df['estimated_diameter_min'].fillna(df['estimated_diameter_min'].median())               #fillning missing value




df['estimated_diameter_max'].fillna(df['estimated_diameter_max'].median())




pd.to_numeric(df['absolute_magnitude'], errors='coerce')                  # Convert necessary columns to numeric




df.drop([ 'name', 'orbiting_body'], axis=1)                           #Remove Unnecessary Columns:

# EDA



plt.figure(figsize=(10, 6))                           #Key Distributions
sns.histplot(df['absolute_magnitude'], kde=True, bins=30)
plt.title('Distribution of Absolute Magnitude')
plt.show()


# In[53]:


print(df.dtypes)


# In[54]:


df_numeric = df.select_dtypes(include=[float, int])                         #Convert data to digital data
print(df_numeric.head())


# In[55]:


plt.figure(figsize=(10, 8))                                                    #Correlation Heatmap  
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)  
plt.title('Correlation Matrix')     
plt.show()   


# In[56]:


print(df.columns)


# In[92]:


print(df.head())                                    #which column contains a date 


# In[82]:


pd.to_datetime(df['name'], errors='coerce')                        #Time Series Analysis


# In[72]:


df['name'].dt.year


# In[73]:


df.groupby('year').size()


# In[81]:


plt.figure(figsize=(12, 6))
neo_per_year.plot()
plt.title('Number of NEOs per Year')
plt.xlabel('Year')
plt.ylabel('Number of NEOs')
plt.grid(True)
plt.show()


# # Data Preprocessing

# In[88]:


features = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']
X = df[features]                                             #Feature Selection


# In[96]:


le = LabelEncoder()
df['orbiting_body_encoded'] = le.fit_transform(df['orbiting_body'])
X.loc[:, 'orbiting_body'] = df['orbiting_body_encoded']


# In[97]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[100]:


imputer = SimpleImputer(strategy='mean')  
X_imputed = imputer.fit_transform(X)


# In[101]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# In[105]:


smote = SMOTE()                                             # Data balancing using SMOTE
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


# In[106]:


print(f"تم معالجة {sum(y_resampled)} عينات خطيرة و {len(y_resampled) - sum(y_resampled)} عينات غير خطيرة")


# In[107]:


scaler = StandardScaler()                                    # Data Scaling
X_scaled = scaler.fit_transform(X_resampled)


# # Model training

# In[110]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)      # Data partitioning                           


# In[5]:


clf = RandomForestClassifier()                         # Create a RandomForestClassifier model

# In[7]:

clf.fit(X_resampled, y_resampled)                     # Training the model on data

# In[ ]:

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))                         # Model Evaluation

# In[ ]: