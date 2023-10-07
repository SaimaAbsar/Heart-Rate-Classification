# Author: Saima Absar

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, LSTM
from keras import optimizers, metrics
from keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

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
print(len(label))
data = np.asarray(data)
#print(data.shape)
data = normalize(data, axis=0, order=1)

#shuffling the set
data,label =shuffle(data, label, random_state=0) 
print(data)
#reshaping the data
data= data.reshape(data.shape[0], data.shape[1], 1)
print(data.shape)


X=data
y=np.asarray(label)

# LSTM model 2

model = Sequential()

model.add(LSTM(units = 70, return_sequences = True, input_shape = (X.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 70, return_sequences = True))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units = 1,activation='sigmoid'))
print(model.summary())


# Optimizer
#sgd = optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)
opt = optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

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
    #print(X_train.shape, y_train.shape)
    model.fit(X_train, y_train, batch_size=64, epochs=200)
    sc, ac = model.evaluate(X_test, y_test, batch_size=None)
    scores.append(sc)
    accuracy.append(ac)
    
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



