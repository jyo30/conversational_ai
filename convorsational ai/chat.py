import random
import json
import pickle
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import smtplib
import json
import numpy as np
import random
import json
import torch
import torch.nn as nn
#importing nltk and necessary downloads
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader
nltk.download('all')
data_file = open('intents.json').read()
intents = json.loads(data_file)
#tokenization
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
#stemming
stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower())
#bag of words
def BagOfWords(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    Bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            Bag[idx] = 1
    return Bag
#tags and pairs,stop words removal
all_the_words = []
tags = []
pair = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_the_words.extend(w)
        pair.append((w, tag))
ignore_words = ['?', '.', '!',',']
all_the_words = [stem(w) for w in all_the_words if w not in ignore_words]
# remove duplicates and sort
all_the_words = sorted(set(all_the_words))
tags = sorted(set(tags))

print(len(pair), "patterns")
print(len(tags), "tags:", tags)
print(len(all_the_words), "unique stemmed words:", all_the_words)
#creating data
X_train = []
y_train = []
for (pattern_sentence, tag) in pair:
    bag = BagOfWords(pattern_sentence, all_the_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print("inputsize=",input_size)
print("outputsize=",output_size)
len(bag)
#training data
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples
#neural network
class NeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetModel, self).__init__()
        self.linearlayer1 = nn.Linear(input_size, hidden_size)
        
        self.bn1= nn.BatchNorm1d(hidden_size)
        
        self.linearlayer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2= nn.BatchNorm1d(hidden_size) 
        
        self.linearlayer3 = nn.Linear(hidden_size, num_classes)
        
        self.relu = nn.ReLU()
    def forward(self, x):
        output = self.linearlayer1(x)
        output = self.relu(output)
        output = self.linearlayer2(output)
        output = self.relu(output)
        output = self.linearlayer3(output)
        return output
#suppose if Gpu is available, we can push our model to the device. otherwise we can have in cpu itself
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetModel(input_size, hidden_size, output_size).to(device)
#counting the no of parameters it have. 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count_parameters(model)
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetModel(input_size, hidden_size, output_size).to(device)
# Loss is CrossEntropyLoss and optimizer is Adam
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total=0
correct=0
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scores = model(words)
        _, pred = scores.max(1)
        total += len(words)
        correct += (pred==labels).sum()
    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f} And Got {correct} / {total} with accuracy {float(correct)/float(total)*100:.2f}')
print(f'final Accuracy-----> Got {correct} / {total} with accuracy {float(correct)/float(total)*100:.2f}') 
print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_the_words,
"tags": tags
}
#storing these  in a file, this will serialise it and save it to that file named bot.pth.
FILE = "bot.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
FILE = "bot.pth" 
data = torch.load(FILE)
#defining values
input_size = data["input_size"] 
hidden_size = data["hidden_size"] 
output_size = data["output_size"] 
all_the_words = data['all_words'] 
tags = data['tags'] 
model_state = data["model_state"]
# print the model
model = NeuralNetModel(input_size, hidden_size, output_size).to(device) 
model.load_state_dict(model_state) 
model.eval()
#bot
#### from fuzzywuzzy import fuzz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from fuzzywuzzy import fuzz

bot_name = "RVBot"
name = input("Enter Your Name: ")
print(f"{bot_name}: Hey, let's chat!")

def track_order_status():
    response = "Your order is currently in transit and will be delivered on March 20th."
    print(response)

def track_order_cancel():
    response = "Your order has been canceled and a refund will be issued to your original payment method."
    print(response)

def inventory_management():
    inventory_data = pd.read_csv("inventory_data.csv")
    inventory_data = inventory_data[inventory_data["Quantity"] > 0]
    print(inventory_data)

def shipping():
    shipping_data = pd.read_csv("shipping_data.csv")
    shipping_data["Date"] = pd.to_datetime(shipping_data["Date"])
    shipping_data["Month"] = shipping_data["Date"].dt.month
    shipping_fig = plt.figure(figsize=(8, 4), dpi=100)
    shipping_ax = shipping_fig.add_subplot(111)
    shipping_ax.bar(shipping_data["Month"], shipping_data["Total"], color="blue")
    shipping_ax.set_xlabel("Month")
    shipping_ax.set_ylabel("Total Shipping Cost")
    shipping_ax.set_xticks(range(1, 13))
    shipping_ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.show()

def display_buttons():
    track_order_button = "1. Track Order\n"
    inventory_management_button = "2. Inventory Management\n"
    shipping_button = "3. Shipping\n"
    exit_button = "4. anyother information\n"
    print("Please select an option:\n" + track_order_button + inventory_management_button + shipping_button + exit_button)

def display_track_order_options():
    track_order_status_button = "1. Check order status\n"
    track_order_cancel_button = "2. Cancel order\n"
    back_button = "3. Back\n"
    print("Please select an option:\n" + track_order_status_button + track_order_cancel_button + back_button)
def predict_sales():
    print("RVBot: I'll help you, please provide the required details.")
    day = input("Please Enter the Day you want to predict: ")
    month = input("Please Enter the Month you want to predict: ")
    year = input("Please Enter the Year you want to predict: ")
    store = input("Enter your store id: ")
    item = input("Enter the item id: ")
    filename = 'salesmodel.pickle'
    loaded_model = pickle.load(open(filename, 'rb'))
    prediction = loaded_model.predict(np.array([[day,month,year,store,item]]))
    ans = round(prediction.tolist()[0], 2)
    print(f"RVBot: Your predicted sale on particular date is {ans}.")
    user_response = input("RVBot: Do you want to continue? (yes/no) ")

print("select any options")

while True:
    display_buttons()
    user_input = input("Enter your choice (1-4): ")
    
    if user_input == "1":
        while True:
            display_track_order_options()
            track_order_choice = input("Enter your choice (1-3): ")
            
            if track_order_choice == "1":
                track_order_status()
                break
                
            elif track_order_choice == "2":
                track_order_cancel()
                break
                
            elif track_order_choice == "3":
                break
                
            else:
                print("Invalid choice. Please enter a number between 1 and 3.")
                
    elif user_input == "2":
        inventory_management()
        
    elif user_input == "3":
        shipping()
        
    elif user_input == "4":
        while True:
            sent = input(f"{name}: ")
            sent_score = fuzz.partial_ratio(sent.lower(), ["sales prediction", "sale"])
            
            if "sale" in sent.lower() or sent_score >= 80:
                predict_sales()
                
                
                if user_response.lower() == 'no':
                    print('RVBot: Still any queries? Happy to help :)')
                    continue
                else:
                    predict_sales()
                    continue
          
                 
            patterns = [intent["patterns"] for intent in intents["intents"]]
            flat_patterns = [pattern for sublist in patterns for pattern in sublist]
            best_score = 0
            best_pattern = ""
            
            for pattern in flat_patterns:
                pattern_score = fuzz.partial_ratio(sent.lower(), pattern.lower())
                
                if pattern_score > best_score:
                    best_score = pattern_score
                    best_pattern = pattern
                    
            if best_score >= 20:
                for intent in intents["intents"]:
                    if best_pattern in intent["patterns"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
                        
            elif sent.lower() == "bye":
                print(f"{bot_name}: Bye!")
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

                if prob.item() > 0.10:
                    for intent in intents['intents']:
                        if tag == intent["tag"]:
                            print(f"{bot_name}: {random.choice(intent['responses'])}")
                            break
                else:
                    print(f"{bot_name}: I do not understand...")

                 