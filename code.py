



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('student-mat.csv',sep=';')
data=data[['G1','G2','G3','studytime','failures','absences']]

X=np.array(data.drop(columns='G3'))
y=np.array(data['G3'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

obj=LinearRegression()
obj.fit(X_train,y_train)
acc=obj.score(X_test,y_test)
print(acc)

print('coefficient is=',obj.coef_)
print('intercept is=',obj.intercept_)

predictions=obj.predict(X_test)
for i in range(len(predictions)):
    print(predictions[i],X_test[i],y_test[i])