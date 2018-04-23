from keras.preprocessing.text import Tokenizer

from learning.lstm import TrainingModel
from learning.data import getTrainData

if __name__ == '__main__':
    model = TrainingModel()
    model.build()
    token = Tokenizer(num_words=150)
    x_train, y_train = getTrainData(token)
    model.load()
    model.train(x_train, y_train, times=1000)
    messages = ["Tell me about your leadership",
                "what are you good at? "]
    result = model.askQuestion(token, messages)
    print(result);
