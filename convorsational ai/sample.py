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

# # Load intents from JSON file
# with open('intent.json', 'r') as json_data:
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


# def modify_response_based_on_context(response, context):
#     if context:
#         for intent in intents['intents']:
#             if 'context' in intent and intent['context'] == context:
#                 response = random.choice(intent['responses'])
#                 break
#     return response
# name = input("Enter Your Name: ")
# print("RVBot: Hey, Let's chat!")
# while True:
#     context = ""
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
#                     if context and 'context' in intent and context not in intent["context"]:
#                         continue
#                     response = random.choice(intent['responses'])
#                     if "context" in intent:
#                         context = intent["context"]
#                     response = modify_response_based_on_context(response, context)
#                     print(f"{bot_name}: {response}")
#         else:
#             print(f"{bot_name}: I do not understand...")





# import random

# # Define the intents
# intents = [
#     {
#         "tag": "greeting",
#         "patterns": ["hi", "hello", "hey"],
#         "responses": ["Hello!", "Hi there!", "Hey! How can I assist you today?"],
#         "context": ["none"]
#     },
#     {
#         "tag": "places_in_singapore",
#         "patterns": ["Marina Bay Sands", "Gardens by the Bay", "Sentosa Island", "Merlion Park", "Botanic Gardens", "Night Safari", "Singapore Flyer"],
#         "responses": ["Sure, what date would you like to visit {}?", "Would you like me to book a ticket to {} on your preferred date?"],
#         "context": ["none"]
#     },
#     {
#         "tag": "places_in_hyderabad",
#         "patterns": ["Charminar", "Golconda Fort", "Chowmahalla Palace", "Hussain Sagar Lake", "Qutb Shahi Tombs", "Mecca Masjid"],
#         "responses": ["Sure, what date would you like to visit {}?", "Would you like me to book a ticket to {} on your preferred date?"],
#         "context": ["none"]
#     },
#     {
#         "tag": "places_to_visit",
#         "patterns": ["places to visit in singapore", "places to visit in hyderabad"],
#         "responses": ["Sure, we recommend visiting {}.", "You may want to check out {} when you're in the area."],
#         "context": ["none"]
#     },
#     {
#         "tag": "book_ticket",
#         "patterns": ["book a ticket"],
#         "responses": ["Sure, to which location would you like to book a ticket?", "Where would you like to travel to?"],
#         "context": ["none"]
#     }
# ]

# # Create a dictionary to store the information about different locations and their corresponding city/country
# locations = {"Marina Bay Sands": "Singapore", "Gardens by the Bay": "Singapore", "Sentosa Island": "Singapore", "Merlion Park": "Singapore", "Botanic Gardens": "Singapore", "Night Safari": "Singapore", "Singapore Flyer": "Singapore", "Charminar": "Hyderabad", "Golconda Fort": "Hyderabad", "Chowmahalla Palace": "Hyderabad", "Hussain Sagar Lake": "Hyderabad", "Qutb Shahi Tombs": "Hyderabad", "Mecca Masjid": "Hyderabad", "Mumbai": "India"}

# # Define the function to generate the bot's response
# def generate_response(message, context):
#     # Iterate through the intents
#     for intent in intents:
#         # Check if the input message matches any of the patterns
#         if any(pattern in message.lower() for pattern in intent["patterns"]):
#             # Check if the context of the intent matches the context of the user's message
#             if context in intent["context"]:
#                 # Check if the user mentions "book a ticket"
#                 if "book a ticket" in message.lower():
#                     # Check if the context is related to a specific location
#                     if context == "places_in_hyderabad" or context == "places_to_visit":
#                         response = random.choice(intent["responses"]).format(locations[intent["patterns"][0]])
#                     else:
#                         response = random.choice(intent["responses"]).format(message.title())
#                 else:
#                     # Select a random response from the intent's responses list
#                     response = random.choice(intent["responses"]).format(locations[intent["patterns"][0]])
#                 return response, intent["context"]
#     # If no matching intent is found, return a default response
#     return "I'm sorry, I don't understand. Can you please rephrase your message?", context


# # Main code block
# if __name__ == "__main__":
#     print("Bot: Hi! How can I assist you today?")
#     context = None
#     while True:
#         # Prompt user to enter their message
#         message = input("You: ")

#         # Get bot's response and update context
#         response, context = generate_response(message, context)

#         # Print bot's response
#         print("Bot:", response)



# intents = [
#     {
#         "tag": "greeting",
#         "patterns": ["hi", "hello", "hey", "good morning", "good evening"],
#         "responses": ["Hello, how can I help you?", "Hi there, how can I assist you today?", "Hello! What can I do for you?"],
#         "context": []
#     },
#     {
#         "tag": "places_to_visit",
#         "patterns": ["what are the popular places to visit in Hyderabad", "what are some tourist attractions in Hyderabad", "can you recommend some places to see in Hyderabad"],
#         "responses": ["Some of the popular tourist attractions in Hyderabad are Charminar, Golconda Fort, and Hussain Sagar Lake."],
#         "context": []
#     },
#     {
#         "tag": "places_in_hyderabad",
#         "patterns": ["what are some famous places in Hyderabad", "can you suggest some good places to visit in Hyderabad"],
#         "responses": ["Some of the famous places to visit in Hyderabad are Charminar, Golconda Fort, Chowmahalla Palace, and Mecca Masjid."],
#         "context": []
#     },
#     {
#         "tag": "book_ticket",
#         "patterns": ["I want to book a ticket", "Can you book a ticket for me", "Book a ticket"],
#         "responses": ["Sure, where do you want to go?", "Which destination would you like to travel to?", "Please let me know your preferred travel destination"],
#         "context": ["none"]
#      }
# ]

# import random

# # Create a dictionary to store the information about different locations and their corresponding city/country
# locations = {"Marina Bay Sands": "Singapore", "Gardens by the Bay": "Singapore", "Sentosa Island": "Singapore", "Merlion Park": "Singapore", "Botanic Gardens": "Singapore", "Night Safari": "Singapore", "Singapore Flyer": "Singapore", "Charminar": "Hyderabad", "Golconda Fort": "Hyderabad", "Chowmahalla Palace": "Hyderabad", "Hussain Sagar Lake": "Hyderabad", "Qutb Shahi Tombs": "Hyderabad", "Mecca Masjid": "Hyderabad", "Mumbai": "India"}

# # Define the function to generate the bot's response
# def generate_response(message):
#     # Iterate through the intents
#     for intent in intents:
#         # Check if the input message matches any of the patterns
#         if message in intent["patterns"]:
#             # Check if the context of the intent matches the context of the user's message
#             if "context" in intent:
#                 context = intent["context"][0]
#                 # Check if the user mentions "book a ticket"
#                 if "book a ticket" in message.lower():
#                     # Check if the context is related to a specific location
#                     if context == "places_in_hyderabad" or context == "places_to_visit":
#                         response = "Would you like me to book a ticket to {}?".format(locations[intent["responses"][0].split(",")[0]])
#                     else:
#                         response = "Would you like me to book a ticket to {} on your preferred date?".format(message.title())
#                 else:
#                     # Select a random response from the intent's responses list
#                     response = random.choice(intent["responses"])
#                 return response
#     return "I'm sorry, I don't understand what you're asking. Can you please try again?"

# # Main code block
# if __name__ == "__main__":
#     print("Bot: Hi! How can I assist you today?")
#     context = None
#     while True:
#         # Prompt user to enter their message
#         message = input("You: ")

#         # Get bot's response and update context
#         response, context = generate_response(message, context)

#         # Print bot's response
#         print("Bot:", response)




# import torch
# import random
# import json
# from textblob import TextBlob
# from dataclasses import dataclass

# # Define a Mood dataclass to store sentiment analysis results
# @dataclass
# class Mood:
#     emoji: str
#     sentiment: float

# # Function to get mood using TextBlob sentiment analysis
# def get_mood(input_text: str, *,threshold: float) -> Mood:
#     sentiment: float = TextBlob(input_text).sentiment.polarity
#     friendly_threshold:float = threshold
#     hostile_threshold: float= -threshold

#     if sentiment >= friendly_threshold:
#         return Mood("positive",sentiment)
#     elif sentiment <= hostile_threshold:
#         return Mood("negative",sentiment)
#     else:
#         return Mood("neutral",sentiment)

# # Load pre-trained model and intents from JSON file
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# with open('both.json', 'r') as json_data:
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

# # Bot name
# bot_name = "RVBot"
# context = {}

# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")

# # Main loop for chatbot
# while True:
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
#         break
#     else:
#         # Perform sentiment analysis on user input
#         mood: Mood = get_mood(sent, threshold=0.3)
#         print(f"{mood.emoji} ({mood.sentiment})")

#         # Tokenize and encode user input
#         sent = tokenize(sent)
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         # Pass encoded input through model to get predicted intent tag
#         output = model(X)
#         _, predicted = torch.max(output, dim=1)
#         tag = tags[predicted.item()]

#         # Check if predicted intent tag meets threshold for confidence level
#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         if prob.item() > 0.75:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     if "context" in intent:
#                         context_key = tuple(intent["context"])
#                         if context_key in context and context[context_key] == intent["context"]:
#                             # Use the same context and output response
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#                         else:
#                             # Update the context and output response
#                             context[context_key] = intent["context"]
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#                     else:
#                         # Output response
#                         response = random.choice(intent['responses'])
#                         print(f"{bot_name}: {response}")
#         else:
#             # Output message indicating low confidence level
#             print(f"{bot_name}: I'm sorry, can you please rephrase that?")



# import torch
# import random
# import json
# from textblob import TextBlob
# from dataclasses import dataclass

# # Define a Mood dataclass to store sentiment analysis results
# @dataclass
# class Mood:
#     emoji: str
#     sentiment: float

# # Function to get mood using TextBlob sentiment analysis
# def get_mood(input_text: str, *,threshold: float) -> Mood:
#     sentiment: float = TextBlob(input_text).sentiment.polarity
#     friendly_threshold:float = threshold
#     hostile_threshold: float= -threshold

#     if sentiment >= friendly_threshold:
#         return Mood("positive",sentiment)
#     elif sentiment <= hostile_threshold:
#         return Mood("negative",sentiment)
#     else:
#         return Mood("neutral",sentiment)

# # Load pre-trained model and intents from JSON file
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# with open('both.json', 'r') as json_data:
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

# # Bot name
# bot_name = "RVBot"
# context = {}

# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")

# # Main loop for chatbot
# while True:
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
#         break
#     else:
#         # Perform sentiment analysis on user input
#         mood: Mood = get_mood(sent, threshold=0.3)
#         print(f"{mood.emoji} ({mood.sentiment})")

#         # Tokenize and encode user input
#         sent = tokenize(sent)
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         # Pass encoded input through model to get predicted intent tag
#         output = model(X)
#         _, predicted = torch.max(output, dim=1)
#         tag = tags[predicted.item()]

#         # Check if predicted intent tag meets threshold for confidence level
#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         if prob.item() > 0.75:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     if "context" in intent:
#                         context_key = tuple(intent["context"])
#                         if context_key in context and context[context_key] == intent["context"]:
#                             # Use the same context and output response
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#                         else:
#                             # Update the context and output response
#                             context[context_key] = intent["context"]
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#                     else:
#                         # Output response
#                         response = random.choice(intent['responses'])
#                         print(f"{bot_name}: {response}")
#                     break
#             else:
#                 if mood.sentiment > 0.3:
#                     print(f"{bot_name}: I'm glad to hear that!")
#                 else:
#                     print(f"{bot_name}: Oh sorry, I didn't understand that. Can you please rephrase it?")
#         else:
#             print(f"{bot_name}: Oh sorry, I didn't understand that. Can you please rephrase it?")





########################sentimental analysis using textblob#####################

# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
# from dataclasses import dataclass
# from textblob import TextBlob

# # Define a Mood dataclass to store sentiment analysis results
# @dataclass
# class Mood:
#     emoji: str
#     sentiment: float

# # Function to get mood using TextBlob sentiment analysis
# def get_mood(input_text: str, *,threshold: float) -> Mood:
#     sentiment: float = TextBlob(input_text).sentiment.polarity
#     friendly_threshold:float = threshold
#     hostile_threshold: float= -threshold

#     if sentiment >= friendly_threshold:
#         return Mood("positive",sentiment)
#     elif sentiment <= hostile_threshold:
#         return Mood("negative",sentiment)
#     else:
#         return Mood("neutral",sentiment)

# # Load pre-trained model and intents from JSON file
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# with open('both.json', 'r') as json_data:
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


# # Bot name
# bot_name = "RVBot"
# context = {}

# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")

# # Main loop for chatbot
# while True:
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
#         break
#     else:
#         # Perform sentiment analysis on user input
#         mood: Mood = get_mood(sent, threshold=0.3)
#         print(f"{mood.emoji} ({mood.sentiment})")

#         # Tokenize and encode user input
#         sent = tokenize(sent)
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         # Pass encoded input through model to get predicted intent tag
#         output = model(X)
#         _, predicted = torch.max(output, dim=1)
#         tag = tags[predicted.item()]

#         # Check if predicted intent tag meets threshold for confidence level
#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         if prob.item() > 0.75:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     if "context" in intent:
#                         context_key = tuple(intent["context"])
#                         if context_key in context and context[context_key] == intent["context"]:
#                             # Use the same context and output response
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#                         else:
#                             # Update the context and output response
#                             context[context_key] = intent["context"]
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#                     else:
#                         # Output response
#                         response = random.choice(intent['responses'])
#                         print(f"{bot_name}: {response}")
#                     break
#             else:
#                 # Fuzzy matching
#                 intent_scores = {}
#                 for intent in intents['intents']:
#                     for pattern in intent['patterns']:
#                         score = fuzz.token_sort_ratio(sent, pattern)
#                         if score > 50:
#                             intent_scores[intent['tag']] = intent_scores.get(intent['tag'], 0) + score
#                 if intent_scores:
#                     tag = max(intent_scores, key=intent_scores.get)
#                     for intent in intents['intents']:
#                         if tag == intent["tag"]:
#                             if "context" in intent:
#                                 context_key = tuple(intent["context"])
#                                 if context_key in context and context[context_key] == intent["context"]:
#                                     # Use the same context and output response
#                                     response = random.choice(intent['responses'])
#                                     print(f"{bot_name}: {response}")
#                                     break
#                                 else:
#                                     # Update the context and output response
#                                     context[context_key] = intent["context"]
#                                     response = random.choice(intent['responses'])
#                                     print(f"{bot_name}: {response}")
#                                     break
#                             else:
#                                 # Output response
#                                 response = random.choice(intent['responses'])
#                                 print(f"{bot_name}: {response}")
#                             break
#                 else:
#                     if mood.sentiment > 0.3:
#                         print(f"{bot_name}: I'm glad to hear that!")
#                     else:
#                         print(f"{bot_name}: Oh sorry, I didn't understand that. Can you please rephrase it?")
#         else:
#             print(f"{bot_name}: Oh sorry, I didn't understand that. Can you please rephrase it?")




###############################sentimental analysis using vader##################

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from dataclasses import dataclass
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Define a Mood dataclass to store sentiment analysis results
@dataclass
class Mood:
    emoji: str
    sentiment: float
# Function to get mood using Vader sentiment analysis
def get_mood(input_text: str, *, threshold: float) -> Mood:
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(input_text)
    sentiment: float = sentiment_scores['compound']
    friendly_threshold: float = threshold
    hostile_threshold: float = -threshold

    if sentiment >= friendly_threshold:
        return Mood("positive", sentiment)
    elif sentiment <= hostile_threshold:
        return Mood("negative", sentiment)
    else:
        return Mood("neutral", sentiment)
# Load pre-trained model and intents from JSON file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('both.json', 'r') as json_data:
    intents = json.load(json_data)
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

# Main loop for chatbot
while True:
    sent = input(name + ': ')
    if sent.lower() == "bye":
        print(f"{bot_name}: Goodbye, {name}!")
        break
    else:
        # Perform sentiment analysis on user input
        mood: Mood = get_mood(sent, threshold=0.3)
        print(f"{mood.emoji} ({mood.sentiment})")

        # Tokenize and encode user input
        sent = tokenize(sent)
        X = BagOfWords(sent, all_the_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        # Pass encoded input through model to get predicted intent tag
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # Check if predicted intent tag meets threshold for confidence level
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if "context" in intent:
                        context_key = tuple(intent["context"])
                        if context_key in context and context[context_key] == intent["context"]:
                            # Use the same context and output response
                            response = random.choice(intent['responses'])
                            print(f"{bot_name}: {response}")
                            break
                        else:
                            # Update the context and output response
                            context[context_key] = intent["context"]
                            response = random.choice(intent['responses'])
                            print(f"{bot_name}: {response}")
                            break
                    else:
                        # Output response
                        response = random.choice(intent['responses'])
                        print(f"{bot_name}: {response}")
                    break
            else:
                # Fuzzy matching
                intent_scores = {}
                for intent in intents['intents']:
                    for pattern in intent['patterns']:
                        score = fuzz.token_sort_ratio(sent, pattern)
                        if score > 50:
                            intent_scores[intent['tag']] = intent_scores.get(intent['tag'], 0) + score
                if intent_scores:
                    tag = max(intent_scores, key=intent_scores.get)
                    for intent in intents['intents']:
                        if tag == intent["tag"]:
                            if "context" in intent:
                                context_key = tuple(intent["context"])
                                if context_key in context and context[context_key] == intent["context"]:
                                    # Use the same context and output response
                                    response = random.choice(intent['responses'])
                                    print(f"{bot_name}: {response}")
                                    break
                                else:
                                    # Update the context and output response
                                    context[context_key] = intent["context"]
                                    response = random.choice(intent['responses'])
                                    print(f"{bot_name}: {response}")
                                    break
                            else:
                                # Output response
                                response = random.choice(intent['responses'])
                                print(f"{bot_name}: {response}")
                            break
                else:
                    if mood.sentiment > 0.3:
                        print(f"{bot_name}: I'm glad to hear that!")
                    else:
                        print(f"{bot_name}: Oh sorry, I didn't understand that. Can you please rephrase it?")
        else:
            print(f"{bot_name}: Oh sorry, I didn't understand that. Can you please rephrase it?")



