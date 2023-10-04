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


# # Set device to GPU if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load intents from JSON file
# with open('email.json', 'r') as json_data:
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

# # Define an empty dictionary to store previous conversation turns and responses
# conversation_history = {}

# # Bot name
# bot_name = "RVBot"
# context = {}

# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")
# while True:
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
#         break
#     else:
#         sent = tokenize(sent)
        
#         # Check if a response has been memorized for this user input
#         sent_str = ' '.join(sent)
#         if sent_str in conversation_history:
#             print(f"{bot_name}: {conversation_history[sent_str]}")
#             continue

#         # No memorized response found, generate a new response
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         output = model(X)
#         _, predicted = torch.max(output, dim=1)

#         tag = tags[predicted.item()]

#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         if prob.item() > 0.75:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     if "context" in intent:
#                         context_key = tuple(intent["context"])
#                         if context_key in context and context[context_key] == intent["context"]:
#                             # Memorize the response for this user input
#                             response = random.choice(intent['responses'])
#                             conversation_history[sent_str] = response
#                             print(f"{bot_name}: {response}")
#                             print("Conversation History Log:", conversation_history)
#                         else:
#                             context[context_key] = intent["context"]
#                             print(f"{bot_name}: {random.choice(intent['responses'])}")
#                     else:
#                         # Memorize the response for this user input
#                         response = random.choice(intent['responses'])
#                         conversation_history[sent_str] = response
#                         print(f"{bot_name}: {response}")
#                         print("Conversation History Log:", conversation_history)
#         else:
#             print(f"{bot_name}: I'm sorry, can you please rephrase that?")



# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('email.json', 'r') as json_data:
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

conversation_history = {}

class ChatbotModel:
    def __init__(self, device, intents, model, tags, all_the_words):
        self.device = device
        self.intents = intents
        self.model = model
        self.tags = tags
        self.all_the_words = all_the_words
        self.conversation_history = {}
        self.context = None

    def generate_response(self, user_input):
        # Tokenize user input
        sent = tokenize(user_input)

        # Check if a response has been memorized for this user input
        sent_str = ' '.join(sent)
        if sent_str in self.conversation_history:
            response = self.conversation_history[sent_str]
            return response

        # Generate a new response
        X = BagOfWords(sent, self.all_the_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.25:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    if "context" in intent:
                        if self.context and self.context != intent["context"]:
                            continue  # Skip intents with different context
                        context_key = tuple(intent["context"])
                        if context_key in self.conversation_history and self.conversation_history[context_key] == intent["context"]:
                            response = random.choice(intent['responses'])
                            self.conversation_history[sent_str] = response
                            self.context = intent["context"]
                            print(f"Conversation History: {self.conversation_history}")
                            return response
                        else:
                            self.conversation_history[context_key] = intent["context"]
                            self.context = intent["context"]
                            print(f"Conversation History: {self.conversation_history}")
                            return random.choice(intent['responses'])
                    else:
                        response = random.choice(intent['responses'])
                        self.conversation_history[sent_str] = response
                        self.context = None
                        print(f"Conversation History: {self.conversation_history}")
                        return response

        # No matching intent found, return a default response
        return "I'm sorry, I don't have the information to answer that."

# Create an instance of the chatbot model
chatbot = ChatbotModel(device, intents, model, tags, all_the_words)

name = input("Enter Your Name: ")
print(f"RVBot: Hey {name}, Let's chat!")

print("RVBot: Hello! How can I assist you today?")
while True:
    sent = input(f"{name}: ")
    if sent.lower() == "bye":
        print(f"RVBot: Goodbye, {name}!")
        break
    else:
        response = chatbot.generate_response(sent)
        print(f"RVBot: {response}")