import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal-length','sepal-width','petal-length','petal-width','Class']
dataset=pd.read_csv(url,names=names)
#print(dataset)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
#print(x)
#print(y)
#y=pd.get_dummies(y,drop_first=True)
#print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
#print(x_train)
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)
#print(x_train)
#print(x_test)
from sklearn.neighbors import KNeighborsClassifier
Classifier=KNeighborsClassifier(n_neighbors=12)
Classifier.fit(x_train,y_train)
y_pred=Classifier.predict(x_test)
#print(y_pred)
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.show()

print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))

