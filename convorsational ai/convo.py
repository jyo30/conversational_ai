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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
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
    quit_button ="5. quit\n"
    print("Please select an option:\n" + track_order_button + inventory_management_button + shipping_button + exit_button+ quit_button)

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
    #user_response= input("RVBot: Do you want to continue? (yes/no) ")

print("select any options")
exit_flag = False
while not exit_flag:
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
            sent_score = fuzz.partial_ratio(sent.lower(), ["sales prediction", "sale","sales update"])
            
            if "sale" in sent.lower() or sent_score >= 80:
                predict_sales()
                
                user_response= input("RVBot: Do you want to continue? (yes/no) ")
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
                exit_flag = True
                #print(f"{bot_name}: Bye!")
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
    elif user_input == "5":              
        break
                 