import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

target = to_categorical(df['diagnosis'])

var = df.columns[1:]
X_train = df[var][:500]
y_train = target[:500]
X_test = df[var][500:]
y_test = target[500:]

y_train = np.array(y_train)

model = Sequential()
model.add(Dense(32, input_shape=(30,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='SGD', loss='categorical_crossentropy',
             metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=50, batch_size=16)