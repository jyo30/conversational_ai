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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# FILE = "bot.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_the_words = data['all_the_words']
# tags = data['tags']
# model_state = data["model_state"]
# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()
# bot_name = "RVBot" 
# name=input("Enter Your Name: ")
# print("RVBot:Hey, Let's chat!")
# while True:
#     sent=input(name+':')
#     if sent == "bye":
#         break
#     else:
#         sent = tokenize(sent)
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         output = model(X)
#         _, predicted = torch.max(output, dim=1)

#         tag = tags[predicted.item()]

#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         if prob.item() > 0.85:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     print(f"{bot_name}: {random.choice(intent['responses'])}")
#         else:
#             print(f"{bot_name}: I do not understand...")


##code that includes context ###

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load intents from JSON file
# with open('both.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Load pre-trained model
# FILE = "bot.pth"
# data = torch.load(FILE)
# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_the_words = data['all_the_words']
# tags = data['tags']
# model_state = data["model_state"]
# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# # Bot name
# bot_name = "RVBot"
# context = ""

# name = input("Enter Your Name: ")
# print("RVBot: Hey, Let's chat!")
# while True:
#     sent = input(name + ': ')
#     if sent == "bye":
#         break
#     else:
#         sent = tokenize(sent)
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         output = model(X)
#         _, predicted = torch.max(output, dim=1)

#         tag = tags[predicted.item()]

#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         if prob.item() > 0.85:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     if context and context not in intent["context"]:
#                         continue
#                     response = random.choice(intent['responses'])
#                     if "context" in intent:
#                         context = intent["context"]
#                     print(f"{bot_name}: {response}")
#         else:
#             print(f"{bot_name}: I do not understand...")






##################both.json code#########################

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('both.json', 'r') as json_data:
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

# Bot name
bot_name = "RVBot"
context = {}

name = input("Enter Your Name: ")
print(f"{bot_name}: Hey {name}, Let's chat!")
while True:
    sent = input(name + ': ')
    if sent.lower() == "bye":
        print(f"{bot_name}: Goodbye, {name}!")
        break
    else:
        sent = tokenize(sent)
        X = BagOfWords(sent, all_the_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if "context" in intent:
                        context_key = tuple(intent["context"])
                        if context_key in context and context[context_key] == intent["context"]:
                            print(f"{bot_name}: {random.choice(intent['responses'])}")
                        else:
                            context[context_key] = intent["context"]
                            print(f"{bot_name}: {random.choice(intent['responses'])}")
                    else:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I'm sorry, can you please rephrase that?")
