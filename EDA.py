#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all packages
import os
import numpy as np
import pandas as pd
import cv2 as cv
from pathlib import Path
import warnings
from skimage.feature import hog
import matplotlib.pyplot as plt
import tqdm
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None


# In[2]:


root = 'C:/Users/shivanshi/Downloads/'
style_file = 'styles.csv'
image_folder = root + '/images/'
print(root+style_file)
styles = pd.read_csv(Path(root+style_file),error_bad_lines=False)


# In[3]:


#style file
print("Style shape: ", str(styles.shape))
styles.head()


# In[4]:


df = pd.read_csv(root + "styles.csv", error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)
df.head(10)


# In[5]:


df.nunique()
#classification possible for gender,masterCategory,season,usage,year


# In[6]:


df.isnull().sum()

#season has 1 missing value --> replace with mode
#productDisplayName has 1 missing value --> replace with mode
#usage has 47 missing values (we have to deal)


# In[7]:


df['productDisplayName'] = df['productDisplayName'].fillna(df['productDisplayName'].mode()[0])
df['season'] = df['season'].fillna(df['season'].mode()[0])
df['year'] = df['year'].fillna(df['year'].mode()[0])


# In[8]:


df.isnull().sum()


# In[9]:


df.dropna(axis = 0, how ='any',inplace = True)
df.isnull().sum()


# In[10]:


#to know which article type is bought more 
plt.figure(figsize=(7,30))
df.articleType.value_counts().sort_values().plot(kind='barh')


# In[11]:


# who shops the most
plt.figure(figsize=(7,20))
df.gender.value_counts().sort_values().plot(kind='pie')
print(df.gender.value_counts(normalize = True))
plt.title("Distribution of articles gender-wise")


# In[12]:


# which season more shopping happens
plt.figure(figsize=(7,7))
df.season.value_counts().sort_values().plot(kind='bar',rot = 1)
plt.title("Distribution of articles season-wise")


# In[13]:


# Among the 4 different seasons who shops the most?
import seaborn as sns
sns.displot(data=df, x="gender", col="season")


# In[14]:


print(df.usage.value_counts())
print()
print(df.groupby(['usage','gender'])['gender'].count())
plt.figure(figsize=(10,7))
sns.countplot(x="usage",hue = "gender", data=df)


# In[15]:


# top 10 basecolours 
import random
plt.figure(figsize=(10,10))
df_colors = df.groupby(["baseColour"]).size().reset_index(name="counts").sort_values(by=["counts"], ascending=False)
df_colors.head(10)
