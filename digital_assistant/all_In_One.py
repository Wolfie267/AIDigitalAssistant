#!/usr/bin/env python
# coding: utf-8

# <hr>

# # &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **AI Digital Assistant Program**

# <hr>

# ![title](files/images/Jarvis.png)

# <hr>

# > <h1>Abilities</h1>
# <br>
# 
# <ol>
# <h2><li>Artificial Intelligence Chatbot</li></h2>
# <h2><li>Object Detection</li></h2>
# <h2><li>Face Detection</li></h2>
# <h2><li>Product Recommendation</li></h2>
# <h2><li>Fake News Detection</li></h2>
# <h2><li>Pneumonia Prediction</li></h2>
# <h2><li>Covid Prediction</li></h2>
# <h2><li>Handwritten Digits Recognition</li></h2>
# </ol>
# <br>

# <hr>

# > <h1>Add-On Capabilities</h1>
# <br>
# 
# <ol>
# <h2><li>Voice Command Input</li></h2>
# <h2><li>Audio + Text Feedback</li></h2>
# <h2><li>Device Controlling</li></h2>
# </ol>
# <br>

# <hr>

# Neccessary Imports

import json
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from warnings import filterwarnings as fw
fw('ignore')


# ## 2. Opening Intents.json file and loading it


File = open('files/intents.json')
data = json.load(File)


# ## 3. Create Word Net Lemmatizer


lemmatizer = WordNetLemmatizer()                                    # Initializing lemmatizer to get 'stem' of words
words, classes, doc_X, doc_Y = [], [], [], []                       # Some declarations


# ## 4. Iterate Intents and extract essentials



for intent in data['intents']:                                      # loop through all the intents
    for pattern in intent['text']:                                  # tokenize each pattern and append tokens to words
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_Y.append(intent['intent'])
    
    if intent['intent'] not in classes:
        classes.append(intent['intent'])                            # add the intent to the classes

# Removes punctuation, lowercase string, lemmatize words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# Sort the words and classes in alpha order and ensure no duplicates
words, classes = sorted(set(words)), sorted(set(classes))


# ## 5. Exploratory Data Analysis


print(words, classes, doc_X, doc_Y, sep='\n\n')


# ## 6. Picklizing Lists for later use



# Use Inspect functions




# <hr>

# ## 7. Refactor data and Extract Training Dataset


training = []
out_empty = [0] * len(classes)

# Creating Words model
for idx, doc in enumerate(doc_X):
    boWords = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        boWords.append(1) if word in text else boWords.append(0)
    
    output_row = list(out_empty)                        # mark the index of class that the current pattern is associated to
    output_row[classes.index(doc_Y[idx])] = 1
    training.append([boWords, output_row])              # add the one hot encoded boWords and associated classes to training 
    

random.shuffle(training)                                # Shuffle the data
training = np.array(training, dtype=object)             # Convert it to an array

# Split the features and target labels
train_X = np.array(list(training[:, 0]))
train_Y = np.array(list(training[:, 1]))


# ## 8. Creation of Neural Networks


# Some Declarations
input_shape, output_shape = (len(train_X[0]),), len(train_Y[0])
epochs = 200

# Addind layers to the neural networks model
model = Sequential()                                                # Creating Sequential model
model.add(Dense(128, input_shape=input_shape, activation="relu"))   # Adding Dense layer
model.add(Dropout(0.5))                                             # Adding Dropout layer
model.add(Dense(64, activation="relu"))                             # Adding Dense layer
model.add(Dropout(0.3))                                             # Adding Dropout layer
model.add(Dense(output_shape, activation = "softmax"))              # Adding Dense layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01, decay = 1e-6), metrics=["accuracy"])



print(model.summary())


# ## 9. Training the Model


model.fit(x = train_X, y = train_Y, epochs = 1000, verbose = 1)


# ## 10. Saving the Model


model.save('Chatbot.model')


# ## 11. Loading the Model


model = tf.keras.models.load_model('Chatbot.model')


# ## 12. Unpickling Files and extracting Words, Classes, Data


import pickle




# ## 13. Tokenize, Lemmatize, Clean, Predict, Respond


def richText(text):
    """tokenize -> lemmatize -> clean"""
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in tokens]



def wordsBag(text, vocabulary): 
    tokens = richText(text)
    bow = [0] * len(vocabulary)
    for tWord in tokens: 
        for idx, word in enumerate(vocabulary):
            if word == tWord: 
                bow[idx] = 1
    return np.array(bow)



def predict_Class(text, vocabulary, labels): 
    bow = wordsBag(text, vocabulary)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)

    return [labels[r[0]] for r in y_pred]



def get_Response(intents_list, intents_json):
    import random 
    intent = intents_list[0]
    list_of_intents = intents_json["intents"]
    for loi in list_of_intents: 
        if loi["intent"] == intent:
            result = random.choice(loi["responses"])
            break
    return result


# ## 14. Method to Fetch City Data using OpenWeathermap API


from bs4 import BeautifulSoup
import requests

def get_City_Data(city:str = 'pune', data:str = 'all'):
	"""Get CITY Data
	================

	Parameters:
	-----------
	1. city:str - The city name for which data is requested
	2. data:str - Values('all', 'day-time', 'weather') returns city data according to this choice
	
	Returns:
	--------
	city data using OpenWeathermap API"""
	
	try:
		city = city.lower().replace(" ", "+") + " weather"
		headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3' }
		res = requests.get(f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8', headers=headers)

		soup = BeautifulSoup(res.text, 'html.parser')

		location, time, info, weather = map(
			lambda id: soup.select(id)[0].getText().strip(),
			('#wob_loc', '#wob_dts', '#wob_dc', '#wob_tm',)
		)

		if data == 'all': return f"City: {location} :: Date-Time: {time} :: Weather: {info}, {weather}°C"
		if data == 'time': return time
		if data == 'weather': return f"{info}, {weather}°C"
	
	except: return 'not found'


# ## 15. Method to get Current Date-Time


from datetime import datetime

DA_Name = 'Jarvis'


def get_DateTime(data:str = 'date-time'):
    """Get Date Time
	================

	Parameters:
	-----------
	1. data:str - Values('date-time', 'accurate-date-time', 'time', 'accurate-time') returns data according to the choice
	
	Returns:
	--------
	date-time data using Date-Time API"""
    months = ('January', 'February', 'March', 'April')
    cDT = datetime.now()                                                # current date-time
    desiredData = {
        'date-time': f"{cDT:%B %d, 20%y, %H:%M}",
        'time': f"{cDT:%H:%M}",
        'accurate-time': f"{cDT:%H : %M : %S.%f}",
        'accurate-date-time': f"{cDT:%B %d, 20%y, %H:%M:%S.%f}"
    }

    return desiredData.get(data.strip(), 'unknown')


# ## 16. Method to Scrap Dictionary


from PyDictionary import PyDictionary

# import google_images_search
import googlesearch as google

import webbrowser as wb
from urllib3 import *
from urllib import * 

from wikipedia import *
from pyjokes import *

import pandas as pd
import sys


dictionary = PyDictionary()
ActionDict = { 'mean' : dictionary.meaning, 'synonym' : dictionary.synonym, 'antonym' : dictionary.antonym, 'translate' : dictionary.translate }


def Access_Dictionary(WordList, Lang = 'en', Action = 'mean'):
    if isinstance(WordList, str): WordList = [WordList]

    try: return list(map(ActionDict.get(Action, 'mean'), WordList))
    except: return "Sorry! Some Error Occurred While Surfing Dictionary"


# ## 17. Method to Surf Google


def SearchGoogle(Query = "Python"):
    try:
    # Website Versatility : Counting how many times Google Recommended each site while Querying
        WebsVers = {}
        
        for Website in google.search(Query, tld='com', lang='en', tbs='0', safe='off', num=2, start=0, stop=2, domains=None, pause=2.0, tpe='', country='', extra_params=None, user_agent=None):  #, Return)
            Website = Website[Website.find(':')+1:]
            
            while True:
                if Website[0] not in [':', '/']: break
                Website = Website[1:]
        
            Host = Website[: min(list(map(lambda x: x if x>=0 else 100, [Website.find(':'), Website.find('/')])))]
            WebsVers[Host] = WebsVers.get(Host, 0) + 1
        
        print(pd.DataFrame(zip(WebsVers.keys(), WebsVers.values()), columns = ["Site", "Count"]))


        Max, MaxSite = 0, ""
        for i,j in WebsVers.items():
            if j > Max: MaxSite, Max = i, j
        wb.open(MaxSite)

    except: print("An Error Occurred while Internet Surfing")


# ## 18. Microphone Input, Speaker Output, Commands' Processing


import pyautogui as ag
import pyttsx3
import speech_recognition as sr
import pywhatkit as pwk
import webbrowser as wb

#region SETUP
DA_Name = 'Jarvis'
listener = sr.Recognizer()
engine  = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
#endregion



def Listener(ackListening:bool = False):
    try:
        with sr.Microphone() as source:
            if ackListening: print(f"{DA_Name} :    I AM LISTENING.....")
            audio = listener.listen(source)
            command = listener.recognize_google(audio, language = 'en-US')
            command = ' '.join(command.lower().split())
            return command
    except: return "nothing heard"



def Speak(Str = "Hi, how can I help"):
    print(f"{DA_Name} : {Str} ......")
    engine.say(Str)
    # engine.runAndWait()
    return


# ## 19. Refactor Model Response


# Computations needed for variables
def refactor_Model_Response(text:str, speakerMessage:str = ""):
    computations = { 
        '<currDateTime>' : get_DateTime('date-time' if speakerMessage.find('accurate') == -1 else 'accurate-date-time'),
        '<currTime>' : get_DateTime('time' if speakerMessage.find('accurate') == -1 else 'accurate-time'),
        '<Person>' : speakerMessage.split()[-1] if len(speakerMessage) else 'There',
        '<DA_Name>': DA_Name
    }
    
    for key, val in computations.items(): text = text.replace(key, val)

    return text


# ## 20. Method to inspect Speaker command and perform action if desired


# Common commands
BasicCmds = { 'open', 'play', 'play song', 'play on youtube', 'connect to', 'press', 'take screenshot', 'search' }
deviceApps = { 'explorer', 'file explorer', 'cmd', 'command prompt', 'shell', 'powershell', 'wmplayer', 'windows media player', 'mspaint', 'paint', 'taskmgr', 'task manager', 'notepad', 'calc', 'calculator' }
websites = {
    'google': 'https://www.google.com',
    'instagram': 'https://www.instagram.com/'
}



def Open(app_site_Name:str):
    """Open App-Website
    ----------------
    Performs open action on app or website"""

    app_site_Name = app_site_Name.lower().strip()
    if app_site_Name in deviceApps:
        ag.hotkey('win', 's')
        ag.typewrite(app_site_Name)
        ag.press('enter')

    else: wb.open(websites.get(app_site_Name, 'https://www.google.com'))




def inspect_Speaker_Cmd(command:str):
    """Recognizes Type of Command
    -----------------------------"""

    command = command.lower().strip()

    # Dictionary Search ->  Word - Meaning - Synonym - Antonym
    if any([ "mean" in command, "synonym" in command, "antonym" in command ]):
        for phrase in ["mean by", "meaning of", "synonym of", "synonyms of", "antonym of", "antonyms of"]:
            if phrase in command:
                Speak(str(Access_Dictionary(command[command.find(phrase) + len(phrase) :].strip())))
                break
        else: Speak(str(Access_Dictionary(command.split()[-1])))


    # Fetching Weather Data for a City
    if pos := command.find('weather') != -1:
        
        if 'get' in command:
            command = command.split()
            city = ' '.join(command[command.index('get')+1 : command.index('weather')])
            Speak(f"Fetching {city} weather")
            return get_City_Data(city)
        
        if pos := command.find('of') != -1:
            city = command[pos+2:]
            Speak(f"Fetching {city} weather")
            return get_City_Data(city)
        
        return "not done"
    

    # If Command contains any of the Reserved Commands
    if any(cmd in command for cmd in BasicCmds):
        if pos := command.find('press') != -1:
            # Clean and extract keys
            keys = command[pos+5:].split()
            ag.hotkey(*keys)

        if pos := command.find('open') != -1:
            app_site_name = command[pos+4:]
            Speak(f"Opening {app_site_name}")
            Open(app_site_name)
               
        if pos := command.find('play song') != -1:
            songName = command[pos+9:]
            Speak(f"Playing song {songName}")
            pwk.playonyt(songName)
        
        if pos := command.find('play') != -1:
            songName = command[pos+4:]
            Speak(f"Playing {songName}")
            pwk.playonyt(songName)
        
        if pos:= command.find('search') != -1:
            query = command[pos+6:]
            Speak(f"Searching Google for {query}")
            SearchGoogle(query)

        return "not done"



    #except: print("Error")


# ## 21. Final Chatbot interface for User with Digital Assistant


# Chatbot
def Chatbot(DA_Name = 'Jarvis'):
    # try:
    # print('Hi, How can I help!')
    Speak()
    try:
        while True:
            command = Listener(True)
            if command == 'stop': break
            while command == 'nothing heard':
                command = Listener(True)
            print(f"You: {command}")
            
            command = inspect_Speaker_Cmd(command)

            if command != 'not done':
                intents = predict_Class(command, words, classes)
                result = get_Response(intents, data)
                result = refactor_Model_Response(result, command)
                Speak(result)
            
    except KeyboardInterrupt: return
    # except Exception: return 'Sorry! Something Went Wrong'


# ## 22. Execution


Chatbot()