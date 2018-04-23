import json

from keras.preprocessing import sequence
from keras.utils import np_utils

def readData():
    train_text = []
    y_train = []
    for i in range(0, 8):
        file_name = "./training_data/quest/" + str(i) + ".json"
        with open(file_name, 'r', encoding = 'utf8') as f:
            data = f.read()
        data = data.replace('\n', '')
        data = json.loads(data)
        for j in data:
            train_text.append(j)
            y_train.append(i)
    return (train_text, y_train)


def getTrainData(token):
    train_text, labels = readData()
    token.fit_on_texts(train_text)
    train_seq = token.texts_to_sequences(train_text)
    x_train = sequence.pad_sequences(train_seq, maxlen=9)
    y_train = np_utils.to_categorical(labels)
    return (x_train, y_train)
