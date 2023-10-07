# Author: Saima Absar

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras import optimizers, metrics
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from keras.utils import to_categorical, normalize

# Read merged data
datapath = "./Data/"
file  = pd.read_csv(datapath + "merged.csv")

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
#print(data)
print(len(label))
data = np.asarray(data)

# Normalize the data
data = normalize(data, axis=0, order=1)
#shuffling the set
data,label =shuffle(data, label, random_state=0) 
print(data)
data= data.reshape(data.shape[0], data.shape[1], 1)
print(data.shape)


data_shape=data.shape[1]
print(data_shape)


#CNN network

model = Sequential()
model.add(Conv1D(filters= 64, kernel_size = 3, input_shape=(data_shape,1)))
#model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2,strides=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.add(Activation("relu"))

print(model.summary())

#Configures the model for training
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


X=data
y=np.asarray(label)

#Cross Validation

from sklearn.model_selection import KFold

scores = []
accuracy = []
pre = []
rec = []
f1 = []
#cv = KFold(n_splits=5, shuffle=False)
cv = KFold(n_splits=5, random_state = 0, shuffle=True)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Validation Index: ", test_index)
    
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    #print(X_train.shape, y_train.shape)
    model.fit(X_train, y_train, batch_size=64, epochs=100)
    sc, ac = model.evaluate(X_test, y_test, batch_size=None)
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


#print(score)
#score, accuracy = model.evaluate(X_test, y_test, batch_size=None, verbose=0)
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



