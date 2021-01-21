# Basic-Chatbot-using-DNN

## Overview:

1)	The data is loaded 
2)	The data is then preprocessed.(this is where tokenization and stemming is done)
3)	The data is saved to a pickle file.
4)	The pickle file is then loaded and the essential data is taken for the neural network(NN) model training.
5)	In the training process the data is passed as numpy arrays( these are in binary form), then passed through a neural network that has 2 hidden layers, and finally the probabilities are classified using regression.
6)	The output response is then selected form the json file and displayed to the user.

---
---

* Install python 3.6
* Check if pip is properly install and functioning 
* Create a json file with tags and answers.
* Import python libraries: tensorflow, tflearn, numpy, and nltk 	

## Code explanations:
---

```python
with open("intents.json") as file:
    data = json.load(file)
```
*This helps in opening and loading a “json” file for the program.*

```python
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
```

### Pre-processing starts from here
```python
for intent in data["intents"]:
    for pattern in intents["pattern"]:
```

*Here we’re looping through the json file and retrieving the necessary data for further processing.* 

```python
        wrds = nltk.words_tokenize(pattern)
```

*Splitting a phrase, sentence, paragraph, or an entire text document into smaller units, such as individual words or terms. Each of these smaller units are called tokens.*

```python
        words.extend(wrds)
```

*The “extend” is done because in the previous statement the nltk creates a list of the tokenized words when “ words_tokenize()” is used, so therefore extending the list is much better than again looping through the list and appending the words individually.*

```python
         docs_x.append(wrds)
        docs_y.append(intent["tag"])
```

*These 2 lists are used during the training process of the model, classifying each of the pattern with a tag. This is done in a way that each pattern corresponds to each tag.* 

```python
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
```

*This part can also be done using a for loop.* 

---
*Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form—generally a written word form.*

*Stemming and removing the duplicate elements*


```python
words = [ stemmer.stem(w.lower()) for w in words if w not in "?"]
```

*First the words are converted to lower case and we also don’t consider all the questions marks if any present.*

```python
words = sorted(list(set(words)))
```
*“set()” this removes if any duplicates present.*

*“list()” this converts everything to a list.*

*“sorted()” this arranges everything in a orderly or sorted way.*

```python
labels = sorted(labels)
```
---
### Preparing the model for training

```python
training = []
output = []

out_empty = [0 for _ in range(len(labels))]	
```
*Create an empty list with “0” equal to the number of labels present.*

```python
for x, doc in enumerate(docs_x):
```

*In this for loop the “x” returns integer for every list or iterable variable passed in the enumerate function as argument.*

```python
    bag = []
```
*A one hot encoding is a representation of categorical variables as binary vectors.This first requires that the categorical values be mapped to integer values.Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.*

```python
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_x[x])] = 1
```

*The “output_row” is an empty list, if the tag in the doc_x list is present in the labels list then “1” is put in the index associated in the labels list in the output_row empty list.*

```python
    training.append(bag)
```
*As the neural network model only understands binary values for the purpose of learning, the bag is appended to the training list.*

```python
    output.append(output_row)

training = np.array(training)
output = np.array(output)
```

*These lists are converted to numpy arrays so that it can be read by tflearn for the neural network.*

```python
with open("data.pickle", "wb") as f:
         pickle.dump((words, labels, training, output),f)
```

*All the essential lists or data is stored/written into a pickle file.*

### Training the model

```python
tensorflow.reset_default_graphs()
```
*This is to delete if any previous data was calculated.*

```python
net = tflearn.input_data(shape=[none, len(training[0])])
```

*training[0] - this is done because each training input length must be the same size.*

```python
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
```

*These are 2 hidden nodes of the neural network with 8 neurons in each layer.Fully Connected layers in a neural networks are those layers where all the inputs from one layer are connected to every activation unit of the next layer.*

```python
net = tflearn.fully_connected(net, len(output), activation="softmax")
```

*This is the output layer, the output size will be same as the numpy output array.*

*The softmax function takes as input a vector z of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers.That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval (0,1), and the components will add up to 1, so that they can be interpreted as probabilities. Furthermore, the larger input components will correspond to larger probabilities.*

```python
net = tflearn.regression(net)
```

*The regression layer is used in TFLearn to apply a regression (linear or logistic) to the provided input.*

*arguments: optimizer, loss, metric, and  learning rate*

* *Note: It usually requires to specify a gradient descent optimizer 'optimizer' that will minimize the provided loss function 'loss' (which calculate the errors).One difference is that with a neural network one typically uses gradient descent, whereas with "normal" linear regression one uses the normal equation if possible (when the number of features isn't too huge).*

```python
model = tflearn.DNN(net)
```
*TFLearn provides a model wrapper 'DNN' that can automatically performs a neural network classifier tasks, such as training, prediction, save/restore, etc.*

```python
try:
    model.load("model.tflearn")
```
*This helps to load a previously trained and saved model.*

```python
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
```

*"fit()" is for training the model with the given inputs (and corresponding training labels).This which will train the model by slicing the data into "batches" of size "batch_size", and repeatedly iterating over the entire dataset for a given number of "epochs".*

```python
def bag_of_words(s, words):
    bag= [0 for _ in range(len(words))]

    s_words = nltk.words_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for sw in s_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = (1)

    return np.array(bag)
```
*One hot encoding done here*

```python
def chat():
    print("start your chat here: (to stop the program type "stop") ")
    while True:
        inp = input("you: ")
        if inp.lower() == "stop":
            break
```
*To enter input and if the user enters “stop” the program would terminate.*

```python
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
```
*This will take the maximum probability value from the NN output/result and is saved in the variable “results_index”*

```python
      tag = labels[results_index]
```
*This will save the tag associated in the json file in the tag variable.*

```python
        #print(tag)

        if results[results_index]> 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["response"]
            print(random.choice(responses))
        else:
            print("didn't understand, try again!!")
```
*The result with the maximum percent is crosschecked with the tags in the json file then a random response is chosen and displayed. If the tag is less than 70%, we ask the user to try again, this is just to make the program more optimal.*

---

```python
chat()
```
---

## Note:

* The first “try” and “except” used was to prevent the program to get started with the preprocessing stage even if the preprocessing has already done with the data.
* The “except” part contains all the preprocessing part, the program will go through this code only if the data hasn’t been preprocessed for the training phase.
* The “try” part contains the pickle file that contains all the data saved after the preprocessing stage.
* This helps the program to run faster and more efficiently.
* The second “try” and “except” used was to load the trained model if the training has already been successful and if not the to go ahead with the training process. 
