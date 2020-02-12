#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[49]:


train = pd.read_csv('D:DS_TriS/titanic_data.csv')
train.head()


# In[50]:


train.info()


# In[51]:


train.describe()


# In[52]:


train.isnull().sum()


# In[53]:


#EDA
sns.countplot(x='Survived',data=train)


# In[54]:


sns.countplot(x='Survived',hue='Sex',data=train)


# In[55]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# In[56]:


sns.heatmap(train.corr())


# In[57]:


train['Age']=train['Age'].fillna(train['Age'].mean())


# In[58]:


train['Embarked']=train['Embarked'].fillna('S')


# In[59]:


X = train.iloc[:, [2, 4, 5, 6, 7, 9,11]]
y = train.iloc[:, 1]


# In[60]:


X.head()


# In[61]:


#DUMMIES
sex = pd.get_dummies(X['Sex'], prefix = 'Sex')
embark = pd.get_dummies(X['Embarked'], prefix = 'Embarked')
passenger_class = pd.get_dummies(X['Pclass'], prefix = 'Pclass')
X = pd.concat([X,sex,embark, passenger_class],axis=1)
X.head()


# In[62]:


sns.boxplot(data= X).set_title("Outlier Box Plot")


# In[63]:


X.columns


# In[64]:


X=X.drop(['Sex','Embarked','Pclass'],axis=1)
X.head()


# In[65]:


X['travel_alone']=np.where((X['SibSp']+X['Parch'])>0,1,0)
X.corr()


# In[66]:


X.head()


# In[67]:


X=X.drop(['SibSp','Parch','Sex_male'],axis=1)
X.head()


# In[68]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)


# In[69]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:, [0,1]] = sc.fit_transform(X_train.iloc[:, [0,1]])
X_test.iloc[:, [0,1]] = sc.transform(X_test.iloc[:, [0,1]])


# In[70]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
classifier.fit(X_train, y_train)


# In[71]:


classifier.score(X_test,y_test)


# In[72]:


classifier.score(X_train,y_train)


# In[73]:


x=X.drop(['Fare','Embarked_C','Pclass_2','travel_alone'],axis=1)
x.head()


# In[74]:


y.head()


# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)


# In[76]:


classifier.fit(X_train, y_train)


# In[77]:


classifier.score(X_test,y_test)


# In[78]:


y_pred = classifier.predict(X_test)
y_pred


# In[79]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()


# In[80]:


model_accuracy,model_standard_deviation


# In[81]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[82]:


confusion_matrix


# In[83]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[84]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
area_under_curve = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % area_under_curve)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




