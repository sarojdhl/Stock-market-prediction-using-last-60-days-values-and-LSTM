import math 
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#reading data
df = web.DataReader('AAPL', data_source = 'yahoo', start ='2012-01-01', end ='2019-12-17')


#create a new data frame with only the close columns
data = df.filter(['Close'])
#convert data to numpy array
dataset=data.values
#get the number of rows to train the model
training_data_len=math.ceil(len(dataset)*.8)



#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


#create the training dataset
#create the scaled training data set
train_data = scaled_data[0:training_data_len,:]
#split into x_train and y_train
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])


 #convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#Build the LSTM model
model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


#compile the model
model.compile('adam','mean_squared_error',)


#train the model
model.fit(x_train,y_train,batch_size=1,epochs=1)


#create the testing dataset
#create a new array containing scaled valuse from index 1543 to 2003
test_data=scaled_data[training_data_len-60: ,:]
#create the data set x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])


#convert the data into a numpy array
x_test=np.array(x_test)
y_test=np.array(y_test)
#reshape the data
x_test= np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


#get the model predicted price values
prediction = model.predict(x_test)
predictions=scaler.inverse_transform(prediction)


#get the rms error 
rmse=np.sqrt(np.mean(predictions-y_test)**2)
print('RMSE: '+ str(rmse))


#plot the data
train = data[:training_data_len]
valid=data[training_data_len:]
valid['prediction']=predictions
#visulize the data
plt.figure(figsize=(12,6))
plt.title('model')
plt.xlabel('date',fontsize=14)
plt.ylabel('us$ close price',fontsize=14)
plt.plot(train['Close'])
plt.plot(valid[['Close','prediction']])
plt.legend(['Train','Valid','Prediction'],loc='lower right')
plt.show()
