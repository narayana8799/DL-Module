import numpy as np
class Svm:
    def __init__(self,lr=0.01,alfa=0.01,n_iters=1000):
        self.lr=lr
        self.alfa=alfa
        self.n_iters=n_iters
        self.w=None
        self.c=None
    def fit(self,X,y):
        labels=np.unique(y)
        yhat=np.where(y<=labels[0],-1,1)
        n_samples,n_features=X.shape
        self.w=np.zeros(n_features)
        self.c=0
        for i in range(self.n_iters):
            for ind,x_i in enumerate(X):
                condition=(yhat[ind]*(np.dot(x_i,self.w)-self.c)-1>=0)
                if condition:
                    self.w=self.w-self.lr*(2*self.alfa*self.w)
                else:
                    self.w=self.w-self.lr*(2*self.alfa*self.w-np.dot(x_i,yhat[ind]))
                    self.c=self.c-self.lr*yhat[ind]
        
    def predict(self,X):
        linearmod=np.dot(X,self.w)-self.c
        ypred=np.where(linearmod>=0,1,0)
        return ypred

#running Model


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/content/drive/MyDrive/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting SVM to the Training set
classifier = Svm()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
tp,fp,fn,tn=cm.ravel()

Ac=(tp+tn)/(tn+tp+fp+fn)


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
