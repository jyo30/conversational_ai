import random
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

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('booking.json', 'r') as json_data:
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


import re

# Replacing the destination
bot_name = "RVBot"
context = {}
context_key = None

name = input("Enter Your Name: ")
print(f"{bot_name}: Hey {name}, Let's chat!")

def replace_placeholders(response, placeholders):
    for key, value in placeholders.items():
        response = response.replace("{" + key + "}", value)
    return response

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
        
        if prob.item() > 0.15:
            matched_intent = None
            for intent in intents['intents']:
                if tag in intent["tag"]:
                    matched_intent = intent
                    break
            
            if matched_intent:
                if "context" in matched_intent:
                    context_key = tuple(matched_intent["context"])
                    if context_key in context and context[context_key] == matched_intent["context"]:
                        response = random.choice(matched_intent['responses'])
                        if isinstance(context[context_key], dict):
                            response = replace_placeholders(response, context[context_key])
                        print(f"{bot_name}: {response}")
                    else:
                        context[context_key] = matched_intent["context"]
                        response = random.choice(matched_intent['responses'])
                        if isinstance(context[context_key], dict):
                            response = replace_placeholders(response, context[context_key])
                        print(f"{bot_name}: {response}")
                else:
                    response = random.choice(matched_intent['responses'])
                    if isinstance(matched_intent, dict):
                        response = replace_placeholders(response, matched_intent)
                    print(f"{bot_name}: {response}")
        else:
            print(f"{bot_name}: I'm sorry, can you rephrase it?")
