# Author: Saima Absar

# ###################################
# Results (best acc=0.925 for 5-fold CV)
# Settings: MinMax normalize; random_state=0 (all); training batch=16; epoch=200; lr = 0.001

'''
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
'''

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import shuffle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


# Read merged data
datapath = "./Data/"
file  = pd.read_csv(datapath + "merged_Aug22.csv")

print(file.shape)
data = []
label = []
for i in range(1,file.shape[1],3):
   # print(i)
    j = i+1
    data.append(file.iloc[:,i])
    if file.iloc[2,j] == "icu":
        label.append(1)
    else:
        label.append(0)

data = np.asarray(data)


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
#data = normalize(data, axis=0, order=1)
#print(data.shape)
#np.savetxt("only_heartrate.csv", data, delimiter=",")

#shuffling the set
data,label =shuffle(data, label, random_state=0) 
#print(data)
#reshaping the data
data= data.reshape(data.shape[0], data.shape[1], 1)
print(data.shape)


# In[10]:


#input data shape for input_shape argument.

data_shape=data.shape[1]
print(data_shape)


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, shuffle = True)
print(np.shape(y_test))
print(np.shape(X_train))

#y_train = np.asarray(y_train)

X=data
y=np.asarray(label)
# In[28]:


#CNN-LSTM network

model = Sequential()
model.add(Conv1D(filters= 64, kernel_size = 3, input_shape=(data_shape,1)))
model.add(MaxPooling1D(pool_size=2,strides=1))
#model.add(Flatten())
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2,strides=1))
#model.add(Flatten())
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2,strides=1))
#model.add(Flatten())
model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2,strides=1))
#model.add(Flatten())

model.add(Conv1D(filters=1024, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2,strides=1))

model.add(LSTM(units = 70, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 70, return_sequences = True))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

print(model.summary())


# Optimizer
opt = optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy', optimizer=opt,metrics = ['accuracy'])
#Cross Validation

from sklearn.model_selection import KFold

scores = []
accuracy = []
pre = []
rec = []
f1 = []
cv = KFold(n_splits=5, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)
    
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train, batch_size=16, epochs=100)
    sc, ac = model.evaluate(X_test, y_test, batch_size=None)
    print('Evaluation acc: ', ac)
    scores.append(sc)
    accuracy.append(ac)
    print("Score: ", sc)
    print("Accuracy:", ac)
    
    #Generates output predictions for the input samples

    predictions = model.predict(X_test, batch_size=None, verbose=0)
    y_pred = (predictions>0.5) * 1
    #y_pred =  np.argmax(predictions, axis=1)
    
    # Print f1, precision, and recall scores
    print(predictions)
    print('Y test: ', y_test)
    print('Y pred: ', y_pred)
    
    pre.append(precision_score(y_test, y_pred , average="macro"))
    rec.append(recall_score(y_test, y_pred , average="macro"))
    f1.append(f1_score(y_test, y_pred , average="macro"))
    

print("Score: ", scores)
print("Accuracy:", accuracy)
print('Precision: ', pre)
print('Recall: ', rec)
print('F1-score: ', f1)


print("Avg Accuracy:", np.mean(accuracy))
print("Avg Precision:", np.mean(pre))
print("Avg Recall:", np.mean(rec))
print("Avg F1-score:", np.mean(f1))
print("Avg Loss:", np.mean(scores))

