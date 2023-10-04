import torch
import random
import json
from model import NeuralNet
from nltk_utils import tokenize, BagOfWords
from textblob import TextBlob
from nltk.stem.porter import PorterStemmer
from nltk_utils import tokenize, stem
# Set device to run model on (CPU or GPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load intents from JSON file
# with open('sentiment.json', 'r') as json_data:
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
# context = {}

# name = input("Enter Your Name: ")
# print(f"{bot_name}: Hey {name}, Let's chat!")
# while True:
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
#         break
#     else:
#         # Get the sentiment score using TextBlob
#         sentiment_score = TextBlob(sent).sentiment.polarity
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
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     if "context" in intent:
#                         context_key = tuple(intent["context"])
#                         if context_key in context and context[context_key] == intent["context"]:
#                             # Get the sentiment score for the user's input
                            
#                             # Modify the chatbot's response based on the sentimental score
#                             if sentiment_score >= 0.5:
#                                 print(f"{bot_name}: {random.choice(intent['positive_responses'])}")
#                             elif sentiment_score <= -0.5:
#                                 print(f"{bot_name}: {random.choice(intent['negative_responses'])}")
#                             else:
#                                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#                         else:
#                             context[context_key] = intent["context"]
#                             print(f"{bot_name}: {random.choice(intent['responses'])}")
#                     else:
#                         print(f"{bot_name}: {random.choice(intent['responses'])}")
#         else:
#             print(f"{bot_name}: I'm sorry, can you please rephrase that?")


##########################sentimental_score#############


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load intents from JSON file
# with open('sentiment.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Load saved model
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
# # Get user's name
# name = input("Enter Your Name: ")

# # Welcome the user
# print(f"{bot_name}: Hey {name}, Let's chat!")

# # Start the conversation
# while True:
#     # Prompt the user for input
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
#         break
#     else:
#         sent = tokenize(sent)
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         # Perform sentiment analysis on the user's input
#         blob = TextBlob(' '.join(sent))
#         polarity = blob.sentiment.polarity

#         # Get model's prediction
#         output = model(X)
#         _, predicted = torch.max(output, dim=1)
#         tag = tags[predicted.item()]
#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]

#         if prob.item() > 0.75:
#             matched_intent = None
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     if "context" in intent:
#                         context_key = tuple(intent["context"])     
#                         if context_key in context and context[context_key] == intent["context"]:
#                             matched_intent = intent
#                             break
#                     else:
#                         matched_intent = intent
#                         break
            
#             if matched_intent is not None:
#                 if "sentiment" in matched_intent and matched_intent["sentiment"] == True:
#                     if polarity > 0.5:
#                         print(f"{bot_name}: {matched_intent['responses']['positive']}")
#                     elif polarity < -0.5:
#                         print(f"{bot_name}: {matched_intent['responses']['negative']}")
#                     else:
#                         print(f"{bot_name}: {matched_intent['responses']['neutral']}")
#                 else:
#                     print(f"{bot_name}: {matched_intent['responses'][0]}")
#             else:
#                 print(f"{bot_name}: I'm sorry, I don't understand.")
#         else:
#             print(f"{bot_name}: I'm sorry, can you please rephrase that?")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load intents from JSON file
# with open('sentiment.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Load saved model
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

# # Get user's name
# name = input("Enter Your Name: ")

# # Welcome the user
# print(f"{bot_name}: Hey {name}, Let's chat!")

# # Start the conversation
# while True:
#     # Prompt the user for input
#     sent = input(name + ': ')
#     if sent.lower() == "bye":
#         print(f"{bot_name}: Goodbye, {name}!")
#         break
#     else:
#         # Tokenize and stem the user's input
#         sent = tokenize(sent)
#         all_the_words += [stem(word) for word in sent]
#         X = BagOfWords(sent, all_the_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         # Perform sentiment analysis on the user's input
#         blob = TextBlob(' '.join(sent))
#         polarity = blob.sentiment.polarity

#         # Get model's prediction
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
#                             # Use the same context
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#                         else:
#                             # Update the context
#                             context[context_key] = intent["context"]
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#                     else:
#                         if "sentiment" in intent and intent["sentiment"] == True:
#                             if polarity > 0.5:
#                                 response = intent['responses']['positive']
#                                 print(f"{bot_name}: {response}")
#                                 break
#                             elif polarity < -0.5:
#                                 response = intent['responses']['negative']
#                                 print(f"{bot_name}: {response}")
#                                 break
#                             else:
#                                 response = intent['responses']['neutral']
#                                 print(f"{bot_name}: {response}")
#                                 break
#                         else:
#                             response = random.choice(intent['responses'])
#                             print(f"{bot_name}: {response}")
#                             break
#         else:
#             print(f"{bot_name}: I'm sorry, I don't understand. Can you please try again?")




############################sentimental analysis#####################33

# from textblob import TextBlob
# from dataclasses import dataclass

# @dataclass
# class Mood:
#     emoji: str
#     sentiment: float

# def get_mood(input_text: str, *,threshold: float) -> Mood:
#     sentiment: float = TextBlob(input_text).sentiment.polarity
#     friendly_threshold:float = threshold
#     hostile_threshold: float= -threshold

#     if sentiment >= friendly_threshold:
#         return Mood("that's great",sentiment)
#     elif sentiment <= hostile_threshold:
#         return Mood("that's sad",sentiment)
#     else:
#         return Mood("netural",sentiment)
# if __name__ == '__main__':
#     while True:
#         text: str = input('text: ')
#         mood: Mood = get_mood(text, threshold=0.3)
#         print(f'{mood.emoji} ({mood.sentiment})')


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
