import random
import json
import pickle
import flask
from flask import Flask, render_template, request, jsonify, session, redirect, Response
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
from flask import Flask, request
from flask_socketio import SocketIO, emit

# Flask app
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('intents.json', 'r') as json_data:
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

@app.route('/chat')
def index():
    return 'Hello, World!'

@app.route('/', methods=['POST'])
def chat():
    message = request.get_json()['message']
    print("Received message:", message) 
    response = generate_response(message)
    return response

def generate_response(message):
    sent = tokenize(message)
    X = BagOfWords(sent, all_the_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.85:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = f"{bot_name}: {random.choice(intent['responses'])}"
                return response
    else:
        response = f"{bot_name}: I do not understand..."
        return response


if __name__ == '__main__':
    # Run Flask app
    app.run(host='4.240.13.10',port=5000,debug=True)

