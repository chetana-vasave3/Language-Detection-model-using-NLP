#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import streamlit as st


# In[2]:


df=pd.read_csv("C:\\Users\\adite\\Downloads\\archive\\Language Detection.csv")


# In[11]:


st.title("Language Detection Tool")


# In[ ]:





# In[3]:


def remove_punctuations(text):
    for pun in string.punctuation:
        text=text.replace(pun,"")
    text=text.lower()
    return (text)


# In[4]:


df.Text=df.Text.apply(remove_punctuations)


# In[6]:


from sklearn.model_selection import train_test_split
X=df.Text
y=df.Language
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=12)


# In[7]:


from sklearn import feature_extraction
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
vec=feature_extraction.text.TfidfVectorizer(ngram_range=(1,2),analyzer="char")
NLP_model=pipeline.Pipeline([("vec",vec),("clf",LogisticRegression())])


# In[8]:


NLP_model.fit(X_train,y_train)


# In[10]:


y_pred_test=NLP_model.predict(X_test)


# In[12]:


text_input=st.text_input("Provide Your Text Input Here : ")
Button_click=st.button("Get Language Name")
if Button_click:
    st.text(NLP_model.predict([text_input]))


# In[ ]:




