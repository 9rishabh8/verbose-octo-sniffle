

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

cancer=datasets.load_breast_cancer()

#print(cancer.feature_names)
#print('y=:',cancer.target_names)

X=cancer.data
y=cancer.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

classes=['malignant', 'benign']

obj=svm.SVC(kernel='linear', C=2)

obj.fit(X_train,y_train)
predict=obj.predict(X_test)

acc=metrics.accuracy_score(y_test,predict)
print(acc)

for i in range(len(predict)):
    print(classes[predict[i]],X_test[i],classes[y_test[i]])