from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence


class TrainingModel:
    def __init__(self):
        self.model = Sequential()

    def build(self):
        self.model.add(Embedding(output_dim=32, input_dim=150, input_length=9))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32))
        self.model.add(Dense(units=256, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=8, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    def load(self):
        try:
            self.model.load_weights("./SavedModel/lstm.h5")
            print("load model success, keep training")
        except:
            print("load model fail, start to train a new model")

    def train(self, x_train, y_train, times):
        self.model.fit(x_train, y_train, batch_size=10, epochs=times, verbose=2, validation_split=0.1)
        scores = self.model.evaluate(x_train, y_train, verbose = 2)
        self.model.save_weights("./SavedModel/lstm.h5")

    #below are some tests
    def askQuestion(self, token, inputMessages):
        ask_seq = token.texts_to_sequences(inputMessages)
        ask_final = sequence.pad_sequences(ask_seq, maxlen=9)
        prediction = self.model.predict_classes(ask_final)
        return prediction
    #you should use json to turn the prediction(which means label) into its inside content


"""
You can get higher val_acc(which means validation_accuracy, it will use 10%(because validation_split)
of train data to test the accuracy while training) by change the train_histroy's batch_size、raise the dataset、
token's num_words、maxlen.
And raise the epochs will train more times, suppose to  makes the loss down and the accuracy higher.
The val_loss should keep going down while training, and finally become stable.
If the dataset is too small, you can add new data by just change a little in every old data,
like turn "Wake up." into "Wake up now.".
Finally, if you want to retrain the whole model, just go to SavedModel file and delete the .h5, done.
"""
