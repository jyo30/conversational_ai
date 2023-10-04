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
# with open('travel.json', 'r') as json_data:
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

# #replacing the destination
# bot_name = "RVBot"
# context = {}

# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")
# destination_list = ["mumbai", "delhi", "kolkata", "chennai", "bangalore", "hyderabad", "ahmedabad", "pune", "surat", "jaipur"]
# while True:
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
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
        # if prob.item() > 0.75:
        #     for intent in intents['intents']:
        #         if tag in intent["tag"]:
        #             if "context" in intent:
        #                 context_key = tuple(intent["context"])
        #                 if context_key in context and context[context_key] == intent["context"]:
        #                     response = random.choice(intent['responses'])
        #                     if isinstance(context[context_key], dict):
        #                         for key in context[context_key]:
        #                             response = response.replace("{"+key+"}", str(context[context_key][key]))
        #                     if "{destination}" in response:
        #                         destination = ""
        #                         for word in reversed(sent):
        #                             if word.lower() in destination_list:
        #                                 destination = word
        #                                 break
        #                         response = response.replace("{destination}", destination)
        #                     print(f"{bot_name}: {response}")

        #                 else:
        #                     context[context_key] = intent["context"]
        #                     if "destination" in context[context_key]:
        #                         # Replace the {destination} placeholder with the requested destination
        #                         destination = ""
        #                         for word in reversed(sent):
        #                             if word.lower() in destination_list:
        #                                 destination = word
        #                                 break
        #                         context[context_key]["destination"] = destination
        #                     response = random.choice(intent['responses'])
        #                     if isinstance(context[context_key], dict):
        #                         for key in context[context_key]:
        #                             response = response.replace("{"+key+"}", str(context[context_key][key]))
        #                     if "{destination}" in response:
        #                         destination = ""
        #                         for word in reversed(sent):
        #                             if word.lower() in destination_list:
        #                                 destination = word
        #                                 break
        #                         response = response.replace("{destination}", destination)
        #                     print(f"{bot_name}: {response}")
        #             else:
        #                 response = random.choice(intent['responses'])
        #                 for key in intent:
        #                     if key != "tag" and key != "responses":
        #                         response = response.replace("{"+key+"}", str(intent[key]))
        #                 if "{destination}" in response:
        #                     destination = ""
        #                     for word in reversed(sent):
        #                         if word.lower() in destination_list:
        #                             destination = word
        #                             break
        #                     response = response.replace("{destination}", destination)
        #                 print(f"{bot_name}: {response}")
        #     print(f"{bot_name}: Is there anything else I can help you with?")
        # else:
        #     print(f"{bot_name}: I'm sorry, can you please rephrase that?")




###dilouge manager##############
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Load intents from JSON file
# with open('travel.json', 'r') as json_data:
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

# def replace_placeholders(text, placeholders):
#     """
#     Replaces placeholders in a given text with their corresponding values.

#     Parameters:
#     text (str): The text to replace placeholders in.
#     placeholders (dict): A dictionary containing the placeholders and their values.

#     Returns:
#     str: The text with the placeholders replaced.
#     """
#     for placeholder, value in placeholders.items():
#         if value:
#             text = text.replace("{" + placeholder + "}", str(value))
#     return text



# # Define dictionary to store context information
# context = {}
# # Define list of placeholder names
# placeholder_names = ["destination", "departure_date", "cabin_type"]
# # Define bot name
# bot_name = "RVBot"
# # Ask for user name
# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")
# while True:
#     # Get user input
#     sent = input(name + ': ')
#     # End conversation if user says "bye"
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
#         break
#     else:
#         # Tokenize user input and convert to bag of words representation
#         sent = tokenize(sent)
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)
#         # Make prediction using pre-trained model
#         output = model(X)
#         _, predicted = torch.max(output, dim=1)
#         tag = tags[predicted.item()]
#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         # If predicted tag has high probability, generate response
#         if prob.item() > 0.05:
#             for intent in intents['intents']:
#                 if tag in intent["tag"]:
#                     if "context" in intent:
#                         # If the intent has context information, update the context and generate a response
#                         context_key = tuple(intent["context"])
#                         if context_key in context and context[context_key] == intent["context"]:
#                             # If the context is the same as the previous intent, generate response using stored values
#                             response = random.choice(intent['responses'])
#                             response = replace_placeholders(response, context[context_key])
#                             if "<input>" in response:
#                                 input_placeholder = response.split("<input>")
#                                 user_input = input(input_placeholder[0] + ": ")
#                                 response = input_placeholder[1].replace("{user_input}", user_input)
#                             print(f"{bot_name}: {response}")
            
#                         else:
#                             # If the context is different from the previous intent, update the context and generate a response
#                             context[context_key] = intent["context"]
#                             placeholders = {}
#                             for placeholder in placeholder_names:
#                                 if placeholder in context[context_key]:
#                                     placeholders[placeholder] = context[context_key][placeholder]
#                                 else:
#                                     placeholders[placeholder] = None
#                             response = random.choice(intent['responses'])
#                             response = replace_placeholders(response, placeholders)
#                             if "<input>" in response:
#                                 input_placeholder = response.split("<input>")
#                                 placeholder_name = input_placeholder[1].strip("{}")
#                                 user_input = input(input_placeholder[0] + ": ")
#                                 placeholders[placeholder_name] = user_input
#                                 response = replace_placeholders(input_placeholder[1], placeholders)
#                                 print(f"{bot_name}: {response}")
#                             else:
#                                 print(f"{bot_name}: {response}")
#         else:
#             print(f"{bot_name}: I'm sorry, can you please rephrase that?")

#####static######
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Load intents from JSON file
# with open('travel.json', 'r') as json_data:
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
# bot_name = "RVBot"
# context = {}

# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")
# destination_list = ["mumbai", "delhi", "kolkata", "chennai", "bangalore", "hyderabad", "ahmedabad", "pune", "surat", "jaipur"]
# while True:
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
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
#         if prob.item() > 0.75:
#             found_matching_intent = False
#             for intent in intents['intents']:
#                 if tag in intent["tag"]:
#                     found_matching_intent = True
#                     if "context" in intent:
#                         context_key = tuple(intent["context"])
#                         if context_key in context and context[context_key] == intent["context"]:
#                             response = random.choice(intent['responses'])
#                             if isinstance(context[context_key], dict):
#                                 for key in context[context_key]:
#                                     response = response.replace("{"+key+"}", str(context[context_key][key]))
#                             if "{destination}" in response:
#                                 destination = ""
#                                 for word in reversed(sent):
#                                     if word.lower() in destination_list:
#                                         destination = word
#                                         break
#                                 response = response.replace("{destination}", destination)
#                             if "{cabin_type}" in response:
#                                 cabin_type = ""
#                                 for word in sent:
#                                     if word.lower() == "economy" or word.lower() == "business":
#                                         cabin_type = word.lower()
#                                         break
#                                 response = response.replace("{cabin_type}", cabin_type)
#                             if "{departure_date}" in response:
#                                 departure_date = ""
#                                 for word in sent:
#                                     if word.lower() == "today" or word.lower() == "tomorrow" or word.lower() == "next week":
#                                         departure_date = word.lower()
#                                         break
#                                 response = response.replace("{departure_date}", departure_date)
#                             print(f"{bot_name}: {response}")
#                         else:
#                             context[context_key] = intent["context"]
#                             if "destination" in context[context_key]:
#                                 # Replace the {destination} placeholder with the requested destination
#                                 destination = ""
#                                 for word in reversed(sent):
#                                     if word.lower() in destination_list:
#                                         destination = word
#                                         break
#                                 context[context_key]["destination"] = destination
#                             if "cabin_type" in context[context_key]:
#                                 # Replace the {cabin_type} placeholder with the requested cabin type
#                                 cabin_type = ""
#                                 for word in sent:
#                                     if word.lower() == "economy" or word.lower() == "business":
#                                         cabin_type = word.lower()
#                                         break
#                                 context[context_key]["cabin_type"] = cabin_type
#                             if "departure_date" in context[context_key]:
#                                 # Replace the {departure_date} placeholder with the requested departure date
#                                 departure_date = ""
#                                 for word in sent:
#                                     if word.lower() == "today" or word.lower() == "tomorrow" or word.lower() == "next week":
#                                         departure_date = word.lower()
#                                         break
#                                 context[context_key]["departure_date"] = departure_date
#                             response = random.choice(intent['responses'])
#                             if isinstance(context[context_key], dict):
#                                 for key in context[context_key]:
#                                     response = response.replace("{"+key+"}", str(context[context_key][key]))
#                             if "{destination}" in response:
#                                 destination = ""
#                                 for word in reversed(sent):
#                                     if word.lower() in destination_list:
#                                         destination = word
#                                         break
#                                 response = response.replace("{destination}", destination)
#                             if "{cabin_type}" in response:
#                                 cabin_type = ""
#                                 for word in sent:
#                                     if word.lower() == "economy" or word.lower() == "business":
#                                         cabin_type = word.lower()
#                                         break
#                                 response = response.replace("{cabin_type}", cabin_type)
#                             if "{departure_date}" in response:
#                                 departure_date = ""
#                                 for word in sent:
#                                     if word.lower() == "today" or word.lower() == "tomorrow" or word.lower() == "next week":
#                                         departure_date = word.lower()
#                                         break
#                                 response = response.replace("{departure_date}", departure_date)
#                                 print(f"{bot_name}: {response}")
#                             if not found_matching_intent:
#                                 print(f"{bot_name}: I'm sorry, I didn't understand that. Can you please rephrase or provide more information?")

#         else:
#             print(f"{bot_name}: I'm sorry, can you please rephrase that?")












################static_replaced_placeholders##############

# Set device to GPU if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load intents from JSON file
# with open('booking.json', 'r') as json_data:
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

# #replacing the destination
# bot_name = "RVBot"
# context = {}

# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")
# destination_list = ["mumbai", "delhi", "kolkata", "chennai", "bangalore", "hyderabad", "ahmedabad", "pune", "surat", "jaipur"]
# while True:
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
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
#         if prob.item() > 0.15:
#             for intent in intents['intents']:
#                 if tag in intent["tag"]:
#                     if "context" in intent:
#                         context_key = tuple(intent["context"])
#                         if context_key in context and context[context_key] == intent["context"]:
#                             response = random.choice(intent['responses'])
#                             if isinstance(context[context_key], dict):
#                                 for key in context[context_key]:
#                                     response = response.replace("{"+key+"}", str(context[context_key][key]))
#                             if "{destination}" in response:
#                                 destination = ""
#                                 for word in reversed(sent):
#                                     if word.lower() in destination_list:
#                                         destination = word
#                                         break
#                                 response = response.replace("{destination}", destination)
#                             if "{departure_date}" in response:
#                                 departure_date = ""
#                                 for word in sent:
#                                     if re.match(r'\d{4}-\d{2}-\d{2}', word):
#                                         departure_date = word
#                                         break
#                                 response = response.replace("{departure_date}", departure_date)
#                             if "{cabin_type}" in response:
#                                 cabin_type = ""
#                                 for word in sent:
#                                     if word.lower() in ["economy", "premium economy", "business", "first class"]:
#                                         cabin_type = word
#                                         break
#                                 response = response.replace("{cabin_type}", cabin_type)
#                             print(f"{bot_name}: {response}")

#                         else:
#                             context[context_key] = intent["context"]
#                             if "destination" in context[context_key]:
#                                 # Replace the {destination} placeholder with the requested destination
#                                 destination = ""
#                                 for word in reversed(sent):
#                                     if word.lower() in destination_list:
#                                         destination = word
#                                         break
#                                 context[context_key]["destination"] = destination
#                             if "{departure_date}" in context[context_key]:
#                                 # Replace the {departure_date} placeholder with the requested departure date
#                                 departure_date = ""
#                                 for word in sent:
#                                     if re.match(r'\d{4}-\d{2}-\d{2}', word):
#                                         departure_date = word
#                                         break
#                                     context[context_key]["departure_date"] = departure_date
#                             response = random.choice(intent['responses'])
#                             if "cabin_type" in context[context_key]:
#                                 # Replace the {cabin_type} placeholder with the requested cabin type
#                                 cabin_type = ""
#                                 for word in sent:
#                                     if word.lower() in ["economy", "premium economy", "business", "first class"]:
#                                         cabin_type = word
#                                         break
#                                 context[context_key]["cabin_type"] = cabin_type
#                             response = random.choice(intent['responses'])
#                             if isinstance(context[context_key], dict):
#                                 for key in context[context_key]:
#                                     response = response.replace("{"+key+"}", str(context[context_key][key]))
#                             if "{destination}" in response:
#                                 destination = ""
#                                 for word in reversed(sent):
#                                     if word.lower() in destination_list:
#                                         destination = word
#                                         break
#                                 response = response.replace("{destination}", destination)
#                             if "{departure_date}" in response:
#                                 departure_date = ""
#                                 for word in sent:
#                                     if re.match(r'\d{4}-\d{2}-\d{2}', word):
#                                         departure_date = word
#                                         break
#                                 response = response.replace("{departure_date}", departure_date)    
#                             if "{cabin_type}" in response:
#                                 cabin_type = ""
#                                 for word in sent:
#                                     if word.lower() in ["economy", "premium economy", "business", "first class"]:
#                                         cabin_type = word
#                                         break
#                                 response = response.replace("{cabin_type}", cabin_type)
#                             print(f"{bot_name}: {response}")
#                     else:
#                         response = random.choice(intent['responses'])
#                         for key in intent:
#                             if key != "tag" and key != "responses":
#                                 response = response.replace("{"+key+"}", str(intent[key]))
#                         if "{destination}" in response:
#                             destination = ""
#                             for word in reversed(sent):
#                                 if word.lower() in destination_list:
#                                     destination = word
#                                     break
#                             response = response.replace("{destination}", destination)
#                         if "{departure_date}" in response:
#                             departure_date = ""
#                             for word in sent:
#                                 if re.match(r'\d{4}-\d{2}-\d{2}', word):
#                                     departure_date = word
#                                     break
#                             response = response.replace("{departure_date}", departure_date)
#                         if "{cabin_type}" in response:
#                             cabin_type = ""
#                             for word in sent:
#                                 if word.lower() in ["economy", "premium economy", "business", "first class"]:
#                                     cabin_type = word
#                                     break
#                             response = response.replace("{cabin_type}", cabin_type)
#                         print(f"{bot_name}: {response}")
            
#         else:
#             print(f"{bot_name}: I'm sorry can you please rephrase that?")

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

#replacing the destination
bot_name = "RVBot"
context = {}
destination=""
context_key=None
name = input("Enter Your Name: ")
print(f"{bot_name}: Hey {name}, Let's chat!")
destination_list = ["mumbai", "delhi", "kolkata", "chennai", "bangalore", "hyderabad", "ahmedabad", "pune", "surat", "jaipur"]
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
        destination =""
        for word in reversed(sent):
            if word.lower() in destination_list:
                destination = word
                break
        if destination:
            context["destination"]=destination
        if prob.item() > 0.15:
            for intent in intents['intents']:
                if tag in intent["tag"]:
                    if "context" in intent:
                        context_key = tuple(intent["context"])
                        if context_key in context and context[context_key] == intent["context"]:
                            response = random.choice(intent['responses'])
                            if isinstance(context[context_key], dict):
                                for key in context[context_key]:
                                    response = response.replace("{"+key+"}", str(context[context_key][key]))
                            if "{destination}" in response:
                               response = response.replace("{destination}", context.get("destination",""))
                            #print(f"{bot_name}:{response}")
                            if "{departure_date}" in response:
                                departure_date = ""
                                for word in sent:
                                    if re.match(r'\d{4}-\d{2}-\d{2}', word):
                                        departure_date = word
                                        break
                                response = response.replace("{departure_date}", departure_date)
                            if "{cabin_type}" in response:
                                cabin_type = ""
                                for word in sent:
                                    if word.lower() in ["economy", "premium economy", "business", "first class"]:
                                        cabin_type = word
                                        break
                                response = response.replace("{cabin_type}", cabin_type)
                            print(f"{bot_name}: {response}")

                        else:
                            context[context_key] = intent["context"]
                            if "destination" in context[context_key]:
                                # Replace the {destination} placeholder with the requested destination
                                
                                context[context_key]["destination"] = context.get("destination","")
                            response = random.choice(intent['responses'])
                            if "{departure_date}" in context[context_key]:
                                # Replace the {departure_date} placeholder with the requested departure date
                                departure_date = ""
                                for word in sent:
                                    if re.match(r'\d{4}-\d{2}-\d{2}', word):
                                        departure_date = word
                                        break
                                    context[context_key]["departure_date"] = departure_date
                            response = random.choice(intent['responses'])
                            if "cabin_type" in context[context_key]:
                                # Replace the {cabin_type} placeholder with the requested cabin type
                                cabin_type = ""
                                for word in sent:
                                    if word.lower() in ["economy", "premium economy", "business", "first class"]:
                                        cabin_type = word
                                        break
                                context[context_key]["cabin_type"] = cabin_type
                            response = random.choice(intent['responses'])
                            if isinstance(context[context_key], dict):
                                for key in context[context_key]:
                                    response = response.replace("{"+key+"}", str(context[context_key][key]))
                            if "{destination}" in response:
                                response = response.replace("{destination}", context.get("destination",""))
                            if "{departure_date}" in response:
                                departure_date = ""
                                for word in sent:
                                    if re.match(r'\d{4}-\d{2}-\d{2}', word):
                                        departure_date = word
                                        break
                                response = response.replace("{departure_date}", departure_date)    
                            if "{cabin_type}" in response:
                                cabin_type = ""
                                for word in sent:
                                    if word.lower() in ["economy", "premium economy", "business", "first class"]:
                                        cabin_type = word
                                        break
                                response = response.replace("{cabin_type}", cabin_type)
                            print(f"{bot_name}: {response}")
                    else:
                        response = random.choice(intent['responses'])
                        for key in intent:
                            if key != "tag" and key != "responses":
                                response = response.replace("{"+key+"}", str(intent[key]))
                        if "{destination}" in response:
                           response = response.replace("{destination}", context.get("destination",""))
                        if "{departure_date}" in response:
                            departure_date = ""
                            for word in sent:
                                if re.match(r'\d{4}-\d{2}-\d{2}', word):
                                    departure_date = word
                                    break
                            response = response.replace("{departure_date}", departure_date)
                        if "{cabin_type}" in response:
                            cabin_type = ""
                            for word in sent:
                                if word.lower() in ["economy", "premium economy", "business", "first class"]:
                                    cabin_type = word
                                    break
                            response = response.replace("{cabin_type}", cabin_type)
                        print(f"{bot_name}: {response}")
            
        else:
            print(f"{bot_name}: I'm sorry can you please rephrase that?")

