#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data Preprocessing and Feature Engineering 


# In[1]:


# Loading Dataset
import pandas as pd 
import numpy as np

df = pd.read_csv('../data/Flight_delay.csv')
df.head()


# In[2]:


# Missing values check
df.isnull().sum().sort_values(ascending=False)


# In[4]:


# Handling missing Values
df = df.dropna(subset=['ArrDelay'])

delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay']
df[delay_cols] = df[delay_cols].fillna(0)


# In[8]:


# Feature Engineering (Time and Flags)
df['DepHour'] = df['DepTime'] // 100

df['is_peak_hour'] = ((df['DepHour'] >= 16) & (df['DepHour'] <= 22)).astype(int)
df['is_weekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)


# In[9]:


# Feature Engineering (Delay Features)
df['total_cause_delay'] = df[
    ['CarrierDelay','WeatherDelay','NASDelay','LateAircraftDelay']
].sum(axis=1)


# In[11]:


# Target Creation
df['is_delayed'] = (df['ArrDelay'] >= 60).astype(int)

df['is_delayed'].value_counts(normalize=True)


# In[13]:


# Droping irrelevant columns 
drop_cols = [
    'FlightNum', 'TailNum', 'Cancelled', 'CancellationCode',
    'Diverted', 'ArrTime', 'CRSArrTime'
]

df = df.drop(columns=drop_cols, errors='ignore')


# In[14]:


# Train Test Split
from sklearn.model_selection import train_test_split

X = df.drop('is_delayed', axis=1)
y = df['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[15]:


# Preprocessing Setup
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])


# In[16]:


# Pipeline + Model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

