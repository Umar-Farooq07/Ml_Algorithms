import numpy as np
import pandas as pd

def logloss(y, ypred):
    return -1*np.mean(y*np.log(ypred)+(1-y)*np.log(1-ypred))




from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
x, y = data.data, data.target

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




class mylogisticreg():
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, xtrain,ytrain,lr = 0.00001,epochs= 100 ):
        self.weights= np.zeros(xtrain.shape[1])
        self.bias = 0
        
        for i in range(epochs):
            reg_pred= np.dot(xtrain,self.weights)+self.bias
            ypred= 1/(1+ np.exp(-1*reg_pred))
            cost = logloss(ytrain,ypred)
            if(i%10== 0):
                print(cost)
            dw= -1*np.dot(xtrain.T,(ytrain-ypred))/xtrain.shape[0]
            db= -1*np.mean(ytrain-ypred)

            self.weights= self.weights-lr*dw
            self.bias= self.bias -lr*db

    def pred(self, xtest):
        z = np.dot(xtest, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))




    

    
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

mylr = mylogisticreg()
sklr = LogisticRegression()

mylr.fit(xtrain,ytrain, lr = 0.4, epochs=1000)
sklr.fit(xtrain,ytrain)

mylr_pred= mylr.pred(xtest)
sklr_pred= sklr.predict(xtest)

print(mylr.weights)
print("\n")
print(sklr.coef_)