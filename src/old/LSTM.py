from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from src.read_name import getTrainData

model = Sequential()

model.add(Embedding(output_dim=32, input_dim=200, input_length=9))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])


x_train, y_train = getTrainData()
train_history = model.fit(x_train, y_train, batch_size=50, epochs=10, verbose=2, validation_split=0.2)
