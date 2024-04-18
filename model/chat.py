import random
import json
import pickle
import numpy
import mysql.connector

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Open and loads necessary data to use with chat
intents = json.loads(open("model/intents.json").read())
words = pickle.load(open("model/words.pkl", "rb"))
classes = pickle.load(open("model/classes.pkl", "rb"))
model = load_model("model/model.h5")


# Conectar a la base de datos
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="Pinche"
)

# Crear un cursor
cursor = conn.cursor()


def clean_up_sentence(sentence):
    spanish_sentence_tokenizer = nltk.data.load(
        'tokenizers/punkt/spanish.pickle')
    sentence_words = spanish_sentence_tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return numpy.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(numpy.array([bow]))[0]

    ERROR_THRESHOLD = 0.25

    res = [[i, r]for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    res.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in res:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break

    return result


# Running bot
print("Bot is running")

while True:
    message = input("")

    ints = predict_class(message)
    res = get_response(ints, intents)

    if "SELECT" in res:
        # # Ejecutar una instrucción SQL

        cursor.execute(res)
        res = cursor.fetchall()
        for recipe in res:
            print(recipe)

    else:
        print(res)

# Cerrar la conexión
conn.close()
