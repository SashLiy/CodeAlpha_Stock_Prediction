#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# In[2]:


data = pd.read_csv('LHCL Historical Data (2).csv')
data.head()


# In[4]:


# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])


# In[5]:


# Sort the dataframe by date in ascending order
data.sort_values(by='Date', inplace=True)


# In[6]:


# Consider only the 'Price' column for prediction
dataset = data[['Price']].values
dataset = dataset.astype('float32')


# In[7]:


# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[8]:


# Split the dataset into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# In[9]:


# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[10]:


# Reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[11]:


# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[12]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[13]:


# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# In[14]:


# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[15]:


# Invert predictions to original scale
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[16]:


# Plot the results
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# In[17]:


testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[18]:


plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[19]:


# Print the predicted values for the test set
print("Predicted prices for the test set:")
print(testPredict)


# In[20]:


# Create a dataframe for predictions
predictions_df = pd.DataFrame({
    'Date': data['Date'].iloc[-len(testPredict):],  # Use the dates corresponding to the test set
    'Predicted_Price': testPredict.flatten()       # Flatten the test predictions
})

# Print the predictions with dates
print("Predicted prices for the test set:")
print(predictions_df)


# In[ ]:




