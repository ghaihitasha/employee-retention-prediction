#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# In[2]:


df_raw = pd.read_csv('hr_employee_churn_data.csv')


# In[3]:


df_raw.shape


# In[4]:


df_raw.info()


# ## Feature Engineering

# In[5]:


df = df_raw.drop(['empid'],axis=1)
df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df['satisfaction_level'].describe()


# In[8]:


df['satisfaction_level'] = df['satisfaction_level'].fillna(df['satisfaction_level'].mean())


# In[9]:


df.isnull().sum()


# ### Categorical Features

# In[10]:


df['salary'].unique()


# In[11]:


salary_dummies = pd.get_dummies(df['salary'], drop_first = True)
salary_dummies


# In[12]:


df = pd.concat([df,salary_dummies],axis=1)
df.head()


# In[13]:


df.drop(['salary'],axis=1,inplace=True)
df.head()


# ## Train and Test Sets

# In[14]:


X = df.drop(labels='left',axis=1)
y = df['left']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[16]:


X_test.shape


# ## Baseline Model

# To do:
# - optimal threshold
# - evaluation function to compare  models among each others
# - add explaination of baseline model

# In[17]:


y_pred = (X_test['satisfaction_level'] < 0.5).astype(int)
y_pred


# In[18]:


print(classification_report(y_test, y_pred))


# ## Model Selection

# In[27]:


model_param = {
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'param': {
            'n_estimators': [10, 50, 100, 130],
            'criterion': ['gini','entropy'],
            'max_depth': range(2,4,1),
            'max_features': ['sqrt','log2',None]
        }
    },
    'XGBClassifier': {
        'model': XGBClassifier(objective='binary:logistic'),
        'param': {
            'learning_rate': [0.5, 0.1, 0.01, 0.001],
            'max_depth': [3,5,10,20],
            'n_estimators': [10,50,100,200]
        }
    }
}


# In[28]:


scores = []
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)
    model_selection.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })


# In[29]:


scores


# ## Model Building

# to-do:
# - replace hardcode values with results from above
# - feature importance graphs

# In[31]:


xgb = XGBClassifier(objective='binary:logistic',learning_rate=0.1,max_depth=20,n_estimators=50)
xgb.fit(X_train,y_train)


# In[33]:


y_pred = xgb.predict(X_test)


# In[34]:


print(classification_report(y_test,y_pred))


# In[37]:


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('True value')
plt.show()


# In[ ]:




