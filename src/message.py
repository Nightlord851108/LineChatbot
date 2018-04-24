import json

from learning.lstm import TrainingModel

def checkType(message, list):
    for i in list:
        for j in list[i]:
            if message.find(j)!=-1:
                return i
    return False

class Message:
    def __init__(self, input):
        self._input = input.lower()
        self.languageList = {
            'C/C++': [' c ', 'c++', 'c/c++'],
            'Java': [' java '],
            'Python': ['python'],
            'Javascript': ['javascript', 'js']
        }
        self.projectList = {
            'GeometrA': ['geometra'],
            'ezScrum': ['ezscrum']
        }
        self.knowledgeList = {
            'Scrum': ['scrum', 'agile'],
            'Pair Programming': ['mob programming', 'pair programming', 'mob/pair programming', 'pair/mob programming']
        }

    def checkInfo(self):
        result1 = checkType(self._input, self.languageList)
        result2 = checkType(self._input, self.projectList)
        result3 = checkType(self._input, self.knowledgeList)
        if result1:
            return ('language', result1)
        elif result2:
            return ('project', result2)
        elif result3:
            return ('knowledge', result3)
        else:
            return ('natural', self._input)

    def getOutput(self):
        type, key = self.checkInfo()
        if type == 'natural':
            from keras.preprocessing.text import Tokenizer
            from learning.data import getTrainData
            train = TrainingModel()
            train.build()
            token = Tokenizer(num_words=150)
            getTrainData(token)
            train.load()
            key = train.askQuestion(token, [key])
        with open('./data/' + type + '.json', 'r') as f:
            data = json.loads(f.read())
        return data[key]
