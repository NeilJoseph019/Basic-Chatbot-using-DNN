import nltk
import nltk
import numpy as np
import tflearn
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()



with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # data loading and pre-processing

    for intent in data["intents"]:
        for pattern in intents["pattern"]:
            wrds = nltk.words_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [ stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # preparing the model for training

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_x[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
         pickle.dump((words, labels, training, output),f)

# training the model

tensorflow.reset_default_graphs()

net = tflearn.input_data(shape=[none, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag= [0 for _ in range(len(words))]

    s_words = nltk.words_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for sw in s_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = (1)

    return np.array(bag)


def chat():
    print("start your chat here: (to stop the program type "stop") ")
    while True:
        inp = input("you: ")
        if inp.lower() == "stop":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        #print(tag)

        if results[results_index]> 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["response"]

            print(random.choice(responses))

        else:
            print("didn't understand, try again!!")

chat()
