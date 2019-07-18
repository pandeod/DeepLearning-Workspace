#### Part 1 Pre-processing

#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#read dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#handle categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableEncoder_X1=LabelEncoder()
X[:,1]=lableEncoder_X1.fit_transform(X[:,1])

lableEncoder_X2=LabelEncoder()
X[:,2]=lableEncoder_X2.fit_transform(X[:,2])

oneHotEncoder_X=OneHotEncoder(categorical_features=[1])
X=oneHotEncoder_X.fit_transform(X).toarray()

#Split data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

### Part 2 Build ANN

#import keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialize ANN
classifier=Sequential()

#adding input and first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=12))

#adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#adding output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compile ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting ANN to Training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#Predict result
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

print(cm)

#pip install ipykernel cloudpickle


