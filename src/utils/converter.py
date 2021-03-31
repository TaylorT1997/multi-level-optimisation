import sys
import xml
import xml.sax
import nltk, re, pprint
from nltk import word_tokenize
import random

class MyHandler(xml.sax.handler.ContentHandler):
    def __init__(self, filtr):
        self.filtr = filtr
        self.in_sentence = False
        self.in_cue = 0
        self.in_xcope = 0
        self.sentence = []
        self.data = []

    def parse(self, f):
        xml.sax.parse(f, self)
        return self.data

    def characters(self, data):
        if self.in_sentence == False:
            return
        data = data.strip()
        if len(data) > 0:
            for token in word_tokenize(data):
                if self.filtr in ["cue", "cuescope"] and self.in_cue > 0:
                    label = "C"
                elif self.filtr in ["scope", "cuescope"] and self.in_xcope > 0:
                    label = "S"
                else:
                    label = "O"
                self.sentence.append((token, label))

    def startElement(self, name, attrs):
        if name == 'cue':
            self.in_cue += 1
        elif name == 'xcope':
            self.in_xcope += 1
        elif name == 'sentence':
            self.in_cue = 0
            self.in_xcope = 0
            self.in_sentence = True
            self.sentence = []

    def endElement(self, name):
        if name == 'cue':
            self.in_cue -= 1
        elif name == 'xcope':
            self.in_xcope -= 1
        elif name == 'sentence':
            self.in_sentence = False
            self.data.append(self.sentence)







if __name__ == "__main__":
    random.seed(123)
    mode = sys.argv[1]
    filtr = sys.argv[2]
    if mode == "train":
        outputfiles = sys.argv[3:5]
        inputfiles = sys.argv[5:]
    else:
        outputfiles = [sys.argv[3]]
        inputfiles = sys.argv[4:]


    data = []
    for path in inputfiles:
        data += MyHandler(filtr).parse(path)

    if mode == "train":
        random.shuffle(data)
        dev_size = int(len(data) / 10.0)
        data_dev = data[:dev_size]
        data_train = data[dev_size:]

        with open(outputfiles[0], 'w') as f:
            for sentence in data_train:
                for token, label in sentence:
                    f.write(token + "\t" + label + "\n")
                f.write("\n")

        with open(outputfiles[1], 'w') as f:
            for sentence in data_dev:
                for token, label in sentence:
                    f.write(token + "\t" + label + "\n")
                f.write("\n")
    elif mode == "test":
        with open(outputfiles[0], 'w') as f:
            for sentence in data:
                for token, label in sentence:
                    f.write(token + "\t" + label + "\n")
                f.write("\n")
    else:
        raise ValueError("Unknown")



