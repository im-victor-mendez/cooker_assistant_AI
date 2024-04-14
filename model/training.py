# Packages
import random
import json
import pickle
import numpy

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizer_v1 import SGD

nltk.download('punkt')

# Minimize [WordNetLemmatizer] use name
lemmatizer = WordNetLemmatizer

# Load intents
intents = json.loads(open("model/intents.json").read())

# Word and letter collections
words = []
classes = []
documents = []
ignore_letters = ["?", "¿", ".", ","]

#
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Separate each word and create a list of that
        word_list = nltk.word_tokenize(pattern, "Spanish")

        # Insert into [words]
        words.append(word_list)

        # Insert into documents with [intent["tag"]] name
        documents.append((word_list, intent["tag"]))

        # Insert into classes non existent [intent["tag"]]
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

print("Training model")
print("Documents:")
print(documents)
print("Train successfully")
