
#importing libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#importing data
digitss=load_digits()

X=digitss.data
y=digitss.target

print('Image Data Shape',digitss.data.shape)
print('Lable Data Shape',digitss.target.shape)

#looking at data
plt.figure(figsize=(20,4))
for index, (image,lable) in enumerate(zip(digitss.data[0:5],digitss.target[0:5])):
    plt.subplot(1,5,index +1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n'% lable, fontsize = 20)

#splitting data and training model
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=2)

from sklearn.linear_model import LogisticRegression

obj=LogisticRegression()
obj.fit(X_train,y_train)

predict=obj.predict(X_test)
acc=obj.score(X_test,y_test)

print(acc)

cm=metrics.confusion_matrix(predict,y_test)


plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt='.3F',linewidths=0.2,square=True,cmap='Dark2_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title='Accuracy score: {0}'.format(acc)
plt.title(all_sample_title,size=15)


index=0
ClassifiedIndex = []
for i,actual in zip(predict,y_test):
    if i == actual:
        ClassifiedIndex.append(index)
        index+=1

plt.figure(figsize=(20,3))
for plotindex,wrong in enumerate(ClassifiedIndex[0:4]):
    plt.subplot(1,4,plotindex+1)
    plt.imshow(np.reshape(X_test[wrong],(8,8)),cmap=plt.cm.Blues_r)
    plt.title('Predicted : {}, Actual : {}'.format(predict[wrong],y_test[wrong]),fontsize=15)

plt.show()