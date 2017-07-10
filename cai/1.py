import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv('C:/Users/caijiawen/Desktop/train.csv')
test = pd.read_csv('C:/Users/caijiawen/Desktop/test.csv')
train.drop(17, inplace=True)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
rate=0.01
#train=pd.double(train)
train=train.astype(float)
test=test.astype(float)
print(train)
for i in range(train.shape[0]):
    print(train.iat[i,1],train.iat[i,2],train.iat[i,3],train.iat[i,4],train.iat[i,5])
for i in range(train.shape[0]):
    
    train.iat[i,1]=train.iat[i,1]*1e-6
    train.iat[i,2]=train.iat[i,2]*1e-2
    train.iat[i,3]=train.iat[i,3]*1e-7
    train.iat[i,4]=train.iat[i,4]*1e-5
    train.iat[i,5]=train.iat[i,5]*1e-3
for i in range(test.shape[0]):
    
    test.iat[i,1]=test.iat[i,1]*1e-6
    test.iat[i,2]=test.iat[i,2]*1e-2
    test.iat[i,3]=test.iat[i,3]*1e-7
    test.iat[i,4]=test.iat[i,4]*1e-5
    test.iat[i,5]=test.iat[i,5]*1e-3
for i in range(train.shape[0]):
    print(train.iat[i,1],train.iat[i,2],train.iat[i,3],train.iat[i,4],train.iat[i,5])
w1,w2,w3,w4,b=0.1,0.1,0.1,0.1,0.1

def loss (x,y) : 
    return np.sum((x-y)**2)
def losss (x,y) : 
    return ((x-y)**2)
for i in range(train.shape[0]):
    print("iter %d error %f" % (i,loss(train['Price per week']*w1+train['Population of city']*w2+train['Monthly income of riders']*w3+train['Average parking rates per month']*w4+b,train['Number of weekly riders'])))
    w11=w1
    w22=w2
    w33=w3
    w44=w4
    w1=w11+rate*2*(train['Number of weekly riders'][i]-(b+train['Price per week'][i]*w11+train['Population of city'][i]*w22+train['Monthly income of riders'][i]*w33+train['Average parking rates per month'][i]*w44)*train['Price per week'][i])
    w2=w22+rate*2*(train['Number of weekly riders'][i]-(b+train['Price per week'][i]*w11+train['Population of city'][i]*w22+train['Monthly income of riders'][i]*w33+train['Average parking rates per month'][i]*w44)*train['Population of city'][i])
    w3=w33+rate*2*(train['Number of weekly riders'][i]-(b+train['Price per week'][i]*w11+train['Population of city'][i]*w22+train['Monthly income of riders'][i]*w33+train['Average parking rates per month'][i]*w44)*train['Monthly income of riders'][i])
    w4=w44+rate*2*(train['Number of weekly riders'][i]-(b+train['Price per week'][i]*w11+train['Population of city'][i]*w22+train['Monthly income of riders'][i]*w33+train['Average parking rates per month'][i]*w44)*train['Average parking rates per month'][i])
    b=b-rate*2*(train['Number of weekly riders'][i]-(b+train['Price per week'][i]*w11+train['Population of city'][i]*w22+train['Monthly income of riders'][i]*w33+train['Average parking rates per month'][i]*w44)*(-1))
    #print(w1,w2,w3,w4,b)
ans=0
# print(test)
for i in range(test.shape[0]):
    ans+=(losss(test.iat[0,1],(test.iat[0,2]*w1+test.iat[0,3]*w2+test.iat[0,4]*w3+test.iat[0,5]*w4+b)))
print("final error %f" % (ans))
for i in range(test.shape[0]):
    print(test.iat[i,1]*1e6)
    test.iat[i,1]=(test.iat[i,2]*w1+test.iat[i,3]*w2+test.iat[i,4]*w3+test.iat[i,5]*w4+b)*1e6
    print(test.iat[i,1])
# print(test)

# def rmse(act, pred):
#     return np.sqrt(((act - pred) ** 2).mean())

# model = LinearRegression().fit(train.iloc[:, 2:], train.iloc[:, 1])
# result = model.predict(test.iloc[:, 2:])
# print('result: ', result)
# print('rmse: %f' % rmse(test.iloc[:, 1], result))
