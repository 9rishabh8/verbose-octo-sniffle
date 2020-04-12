
#importing libraries

import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing data set
company_data= pd.read_csv('1000_Companies.csv')
X=company_data.iloc[:,0:4].values
y=company_data.iloc[:,4].values

#data visualisation
sns.heatmap(company_data.corr())
plt.show()

#encoding non numerical data
obj=LabelEncoder()
X[:,3]=obj.fit_transform(X[:,3])

obj1=OneHotEncoder(categorical_features = [3], handle_unknown='ignore')
X=obj1.fit_transform(X).toarray()

#removing dummy variable
X=X[:,1:]

#splitting dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#fitting our model with data
obj2=LinearRegression()
obj2.fit(X_train,y_train)

y_pred=obj2.predict(X_test)

acc=obj2.score(X_test,y_test)
print(acc)

for i in range(len(y_pred)):
    print(y_pred[i],X_test[i],y_test[i])