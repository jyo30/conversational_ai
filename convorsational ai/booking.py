import random
import json
import pickle
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import BagOfWords, tokenize
import smtplib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from fuzzywuzzy import fuzz
import warnings
from collections import defaultdict
import datetime
import os
import json
import re
import random
# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('travel.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained model
FILE = "bot.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_the_words = data['all_the_words']
tags = data['tags']
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "RVBot"

context = {}

def respond(sent):
    global context
    
    # load the intents file
    with open("travel.json") as file:
        intents = json.load(file)
        
    # preprocess the input sentence
    sent = tokenize(sent)
    
    # predict the class of the input sentence using the trained model
    X = BagOfWords(sent, all_the_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    prob = torch.softmax(output, dim=1)[0][predicted.item()]

    # if the predicted probability is high enough, choose a response from the intents file
    # if the predicted probability is high enough, choose a response from the intents file
    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if "context" in intent:
                    context_key = tuple(intent["context"])
                    if context_key in context and context[context_key] == intent["context"]:
                        response = random.choice(intent['responses'])
                        if "{" in response and "}" in response:
                            placeholder = response.split("{")[1].split("}")[0]
                            words = " ".join(sent).split()
                            for word in words:
                                if placeholder in word:
                                    response = response.replace("{"+placeholder+"}", word)
                                    break
                        print(f"{bot_name}: {response}")
                    else:
                        context[context_key] = intent["context"]
                        response = random.choice(intent['responses'])
                        if "{" in response and "}" in response:
                            placeholder = response.split("{")[1].split("}")[0]
                            words = " ".join(sent).split()
                            for word in words:
                                if placeholder in word:
                                    response = response.replace("{"+placeholder+"}", word)
                                    break
                        print(f"{bot_name}: {response}")
                else:
                    response = random.choice(intent['responses'])
                    if "{" in response and "}" in response:
                        placeholder = response.split("{")[1].split("}")[0]
                        words = " ".join(sent).split()
                        for word in words:
                            if placeholder in word:
                                response = response.replace("{"+placeholder+"}", word)
                                break
                    print(f"{bot_name}: {response}")
    else:
        print(f"{bot_name}: I'm sorry can you please rephrase that?")

name = input("Enter Your Name: ")
while True:
    sent = input(name + ': ')
    if sent.lower() == "bye":
        print(f"{bot_name}: Goodbye, {name}!")
        break
    respond(sent)

