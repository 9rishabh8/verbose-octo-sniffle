

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

Car_Data=pd.read_csv('car.data')

obj=preprocessing.LabelEncoder()

buying=obj.fit_transform(list(Car_Data['buying']))
maint=obj.fit_transform(list(Car_Data['maint']))
door=obj.fit_transform((list(Car_Data['door'])))
lug_boot=obj.fit_transform(list(Car_Data['lug_boot']))
safety=obj.fit_transform(list(Car_Data['safety']))
cls=obj.fit_transform(list(Car_Data['class']))

predict='class'

X=list(zip(buying,maint,door,lug_boot,safety))
y=list(cls)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

obj2=KNeighborsClassifier(n_neighbors=11)

obj2.fit(X_train,y_train)
acc=obj2.score(X_test,y_test)
print(acc)

predictions=obj2.predict(X_test)
names=['unacc','acc','good','vgood']

for i in range(len(predictions)):
    print('Predicted=',names[predictions[i]],'Data=',X_test[i],'Actual=',names[y_test[i]])
    n=obj2.kneighbors([X_test[i]],9,True)
    print('N:',n)