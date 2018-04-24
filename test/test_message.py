import unittest

from src.message import Message

class MessageTestSuite(unittest.TestCase):
    def setUp(self):
        self.language = 'C++'
        self.project = 'GeometrA'
        self.knowledge = 'scrum'
        self.natural = 'Who are you? '

    def testConstruct(self):
        message1 = Message(self.language)
        self.assertEqual('language', message1.checkInfo()[0])
        message2 = Message(self.project)
        self.assertEqual('project', message2.checkInfo()[0])
        message3 = Message(self.knowledge)
        self.assertEqual('knowledge', message3.checkInfo()[0])
        message4 = Message(self.natural)
        self.assertEqual('natural', message4.checkInfo()[0])

    def testGetOutput(self):
        message = Message(self.language)
        self.assertEqual("I'm currently TA of object-oriented programming course, using C++ language, of Taipei Tech. ",
                        message.getOutput())
