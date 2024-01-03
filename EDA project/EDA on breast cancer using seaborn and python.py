#!/usr/bin/env python
# coding: utf-8

# In[42]:


#importing the essential files
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore a specific warning
warnings.filterwarnings("ignore", category=UserWarning)

data=pd.read_csv(r"D:\datascience_projects\EDA project\data.csv") #loading the csv file used for analysis
data.head(5)#loading the first five entires
cols=data.columns
#print(cols) #printing the columns for better look

#defing the target and independent variables
y=data.diagnosis #defining the target variable
drop_cols= ['Unnamed: 32','id','diagnosis'] #dropping colums that are not relevent for analysis
x=data.drop(drop_cols,axis=1) #creating the independent variable set and since we are dropping cols axis is set to 1 
                              #for row the axis is given as 0
x.head(6)    

#plot diagnonsis distribution
ax=sns.countplot(y,label="count")
B,M=y.value_counts() #storing the individual counts of each binary values B and M
print("No. of Benign Tumors(non-cancerous): ", B)
print("No. of Malignant tumors(cancerous):",M)
x.describe() #descriptive statistic of the independent features

#standardizing the values of the independent features
#after looking at the descriptive stats of the features we can see quite big differences in values,which can lead
#uneven weights,and reducing the accuracy of the predictive model,hence the values are standardied to increase the accuracy
scale=StandardScaler()
std_x=scale.fit_transform(x)
#print(std_x)
std_df=pd.DataFrame(std_x,columns=x.columns)
print(std_df)

#first 10 features 
data=pd.concat([y,std_df.iloc[:,0:10]], axis=1)
data=pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis',data=data,split=True,inner='quart')
plt.xticks(rotation=45)
#next 10 features plot
data=pd.concat([y,std_df.iloc[:,10:20]], axis=1)
data=pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis',data=data,split=True,inner='quart')
plt.xticks(rotation=45)
#next 10 features plot
data=pd.concat([y,std_df.iloc[:,20:30]], axis=1)
data=pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis',data=data,split=True,inner='quart')
plt.xticks(rotation=45)

#after looking at the violin plots of all the features we can see how some of the features are similar to one another
#which could result in problmes while classifications,in order to understand it better we will see the co-relation btw 
#the feature concavity_worst and concave points_worst

#Using joint plot for feature comparision
sns.jointplot(x.loc[:,'concavity_worst'],
             x.loc[:,'concave points_worst'],
             kind='reg',
             color='red')
#the joint plot tells about the degree of correlation between two features and help understand the relatonship btw them.

#observing the values of distributions and their variance using swarm plots
#first 10 features 
data=pd.concat([y,std_df.iloc[:,0:10]], axis=1)
data=pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=45)
#next 10 features plot
data=pd.concat([y,std_df.iloc[:,10:20]], axis=1)
data=pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=45)
#next 10 features plot
data=pd.concat([y,std_df.iloc[:,20:30]], axis=1)
data=pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=45)

#what swarmplot does, is that it stacks the similar values or datapoints of a feature didvided over the category ,on top of
#each other.It tells the sepratability of the features, also easily see the varience accross each column. This helps in 
#understanding which feature can best help in classifying the values into the binary category

#observing pair-wise correlation with a correlation matrix with a heat map overlaid on it
#creating a subplot first: a subplot refers to a grid of smaller, individual plots within a larger figure. It allows you 
#to create multiple plots or visualizations side by side, making it easier to compare and analyze different aspects of your
#data.

f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(x.corr(),annot=True,linewidth=5,fmt='.1f',ax=ax)

#after this we can see the correlation matrix, were the light shaded cells represent highly corelated features,while the 
#darkest shade represents that the features are not corelated and the mediocre ones represent there exist some level of 
#correlation btw the features. In the highly correlated features, we can drop one of the features from the pair
#that is in high level of correlation,Since there exist some level of similarity btw the two which could lead to errors 
#while classification


# In[ ]:





# In[ ]:




