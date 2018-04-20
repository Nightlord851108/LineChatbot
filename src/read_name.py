import json

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from pprint import pprint

def readData():
    train_text = []
    y_train = []
    for i in range(0, 8):
        file_name = "./training_data/quest/" + str(i) + ".json"
        with open(file_name, 'r') as f:
            data = f.read()
        data = data.replace('\n', '')
        data = json.loads(data)
        for j in data:
            train_text.append(j)
            y_train.append(i)
    return (train_text, y_train)


def getTrainData():
    train_text, y_train = readData()
    token = Tokenizer(num_words=200)
    token.fit_on_texts(train_text)
    pprint(len(token.word_index))
    train_seq = token.texts_to_sequences(train_text)
    x_train = sequence.pad_sequences(train_seq, maxlen=9)

    return (x_train, y_train)
