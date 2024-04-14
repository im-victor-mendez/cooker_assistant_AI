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

print("Training model")

# Minimize [WordNetLemmatizer] use name
lemmatizer = WordNetLemmatizer

# Load intents
intents = json.loads(open("model/intents.json").read())

# Word and letter collections
words = []
classes = []
documents = []
ignore_letters = ["?", "¿", ".", ","]

# Train intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Separate each word and create a list of that
        word_list = nltk.word_tokenize(pattern, "Spanish")

        # Add into [words]
        words.extend(word_list)

        # Insert into documents with [intent["tag"]] name
        documents.append((word_list, intent["tag"]))

        # Insert into classes non existent [intent["tag"]]
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Clean letters to be ignored
words = [lemmatizer.lemmatize(word)
         for words in words if word not in ignore_letters]

# Sort a new collection of only one word of each word from [words]
words = sorted(set(words))

# Sort a new collection of only one class of each class from [classes]
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(words, open("classes.pkl", "wb"))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = numpy.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd, metrics=["accuracy"])

model.fit(numpy.array(train_x), numpy.array(
    train_y), epochs=200, batch_size=5, verbose=1)
model.save("model/model.model")

print("Documents:")
print(documents)

print("Train successfully")
print("Done")
