import numpy as np

class mylr():

    def __init__(self):
        self.weights = None
        self.bias= None
    
    def fit(self,xtrain,ytrain,lr,epochs):
        self.weights= np.zeros(xtrain.shape[1])
        self.bias = 0

        for i in range(epochs):
            pred = np.dot(xtrain,self.weights.T)+self.bias
            loss= ytrain-pred
            cost = np.mean(loss**2)

            dw=np.zeros(xtrain.shape[1])
            db = -2*np.mean(loss)
            for j in range(xtrain.shape[1]):
                dw[j] = -2*np.dot(xtrain[:,j],loss.T)/xtrain.shape[0]
            self.weights = self.weights-lr*dw
            self.bias = self.bias-lr*db
            




    def predict(self,xtest):
        return np.dot(xtest,self.weights.T)+self.bias



from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = load_diabetes()

x = data.data
y = data.target

scaler = StandardScaler()
x= scaler.fit_transform(x)

xtrain, xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

model1= mylr()

model1.fit(xtrain,ytrain, 0.005,10000)
print(model1.weights)
model1pred= model1.predict(xtest)



model3= LinearRegression()
model3.fit(xtrain,ytrain)
model3pred= model3.predict(xtest)

print(mean_squared_error(ytest,model1pred),mean_squared_error(ytest,model3pred))





