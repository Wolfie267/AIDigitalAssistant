#!/usr/bin/env python
# coding: utf-8

# <hr>

# # &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **AI Digital Assistant Program**

# <hr>

# ![title](Jarvis.png)

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
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pyautogui as ag
import pywhatkit as pwk
import webbrowser as wb
from warnings import filterwarnings as fw
fw('ignore')



class build_Train_Fit:

    def __init__(self, intentsFile:str = 'intents.json') -> None:
        # Opening Intents.json file and loading it
        try:
            self.data = json.load(open(intentsFile))
            self.lemmatizer = WordNetLemmatizer()                                    # Initializing lemmatizer to get 'stem' of words
        except: raise Exception(f"Cannot open {intentsFile}")
    


    def extract_clean_Data(self):
        # Create Word Net Lemmatizer
        self.words, self.classes, self.doc_X, self.doc_Y = [], [], [], []             # Some declarations

        # Iterate Intents and extract essentials
        for intent in self.data['intents']:                                      # loop through all the intents
            for pattern in intent['text']:                                  # tokenize each pattern and append tokens to words
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                self.doc_X.append(pattern)
                self.doc_Y.append(intent['intent'])
            
            if intent['intent'] not in self.classes:
                self.classes.append(intent['intent'])                            # add the intent to the classes

        # Removes punctuation, lowercase string, lemmatize words
        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in string.punctuation]

        # Sort the words and classes in alpha order and ensure no duplicates
        self.words, self.classes = sorted(set(self.words)), sorted(set(self.classes))



    def prepare_Train_Data(self):
        # Refactor data and Extract Training Dataset

        training = []
        out_empty = [0] * len(self.classes)

        # Creating Words model
        for idx, doc in enumerate(self.doc_X):
            boWords = []
            text = self.lemmatizer.lemmatize(doc.lower())
            for word in self.words:
                boWords.append(1) if word in text else boWords.append(0)
            
            output_row = list(out_empty)                        # mark the index of class that the current pattern is associated to
            output_row[self.classes.index(self.doc_Y[idx])] = 1
            training.append([boWords, output_row])              # add the one hot encoded boWords and associated classes to training 
            

        random.shuffle(training)                                # Shuffle the data
        training = np.array(training, dtype=object)             # Convert it to an array

        # Split the features and target labels
        self.train_X = np.array(list(training[:, 0]))
        self.train_Y = np.array(list(training[:, 1]))

        return self.train_X, self.train_Y
    


    def build_Neural_Networks(self) -> Sequential:
        # 7. Creation of Neural Networks

        # Some Declarations
        input_shape, output_shape = (len(self.train_X[0]),), len(self.train_Y[0])
        epochs = 200

        # Addind layers to the neural networks model
        self.model = Sequential()                                                # Creating Sequential model
        self.model.add(Dense(128, input_shape=input_shape, activation="relu"))   # Adding Dense layer
        self.model.add(Dropout(0.5))                                             # Adding Dropout layer
        self.model.add(Dense(64, activation="relu"))                             # Adding Dense layer
        self.model.add(Dropout(0.3))                                             # Adding Dropout layer
        self.model.add(Dense(output_shape, activation = "softmax"))              # Adding Dense layer

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01, decay = 1e-6), metrics=["accuracy"])
        
        return self.model



    def train(self):
        # Training the Model
        self.model.fit(x = self.train_X, y = self.train_Y, epochs = 1000, verbose = 1)



    def saveFiles(self, modelPath:str = 'files/Chatbot.model', picklePath:str = 'files/pickleFiles'):
        # Saving the Model
        self.model.save(modelPath)
        self.picklize(self.words, 'words')
        self.picklize(self.classes, 'classes')
        self.picklize(self.data, 'data')



    def picklize(Object:list, fileName:str):
        with open(f'files/pickleFiles/{fileName}', 'wb') as fp: pickle.dump(Object, fp)










from capabilities import get_DateTime, get_City_Data, Access_Dictionary, SearchGoogle
from communicate import audio_IO


class chatbot:
    VC = audio_IO()
    
    def __init__(self, assistant_name:str = "Jarvis") -> None:
        self.DA_Name = assistant_name



    # Use Inspect functions
    def unpicklize(fileName:str):
        with open(f'files/pickleFiles/{fileName}', 'rb') as fp: return pickle.load(fp)



    def loadFiles(self, modelPath:str = 'files/Chatbot.model') -> Sequential:
        # Loading the Model
        self.model = tf.keras.models.load_model(modelPath)
        self.words = self.unpicklize('words')
        self.classes = self.unpicklize('classes')
        self.data = self.unpicklize('data')



    # 11. Tokenize, Lemmatize, Clean, Predict, Respond
    def richText(self, text):
        """tokenize -> lemmatize -> clean"""
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(word) for word in tokens]



    def wordsBag(self, text, vocabulary): 
        tokens = self.richText(text)
        bow = [0] * len(vocabulary)
        for tWord in tokens: 
            for idx, word in enumerate(vocabulary):
                if word == tWord: 
                    bow[idx] = 1
        return np.array(bow)



    def predict_Class(self, text, vocabulary, labels): 
        bow = self.wordsBag(text, vocabulary)
        result = self.model.predict(np.array([bow]))[0]
        thresh = 0.2
        y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
        y_pred.sort(key=lambda x: x[1], reverse=True)

        return [labels[r[0]] for r in y_pred]



    def get_Response(self, intents_list, intents_json):
        import random 
        intent = intents_list[0]
        list_of_intents = intents_json["intents"]
        for loi in list_of_intents: 
            if loi["intent"] == intent:
                result = random.choice(loi["responses"])
                break
        return result






    # Refactor Model Response
    # Computations needed for variables
    def refactor_Model_Response(text:str, speakerMessage:str = ""):
        computations = { 
            '<currDateTime>' : get_DateTime('date-time' if speakerMessage.find('accurate') == -1 else 'accurate-date-time'),
            '<currTime>' : get_DateTime('time' if speakerMessage.find('accurate') == -1 else 'accurate-time'),
            '<Person>' : speakerMessage.split()[-1] if len(speakerMessage) else 'There',
            '<DA_Name>': chatbot.VC.DA_Name
        }
        
        for key, val in computations.items(): text = text.replace(key, val)

        return text






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
        if app_site_Name in chatbot.deviceApps:
            ag.hotkey('win', 's')
            ag.typewrite(app_site_Name)
            ag.press('enter')

        else: wb.open(chatbot.websites.get(app_site_Name, 'https://www.google.com'))



    # Method to inspect Speaker command and perform action if desired
    def inspect_Speaker_Cmd(command:str):
        """Recognizes Type of Command
        -----------------------------"""

        command = command.lower().strip()

        # Dictionary Search ->  Word - Meaning - Synonym - Antonym
        if any([ "mean" in command, "synonym" in command, "antonym" in command ]):
            for phrase in ["mean by", "meaning of", "synonym of", "synonyms of", "antonym of", "antonyms of"]:
                if phrase in command:
                    chatbot.VC.speak(str(Access_Dictionary(command[command.find(phrase) + len(phrase) :].strip())))
                    break
            else: chatbot.VC.speak(str(Access_Dictionary(command.split()[-1])))


        # Fetching Weather Data for a City
        if pos := command.find('weather') != -1:
            
            if 'get' in command:
                command = command.split()
                city = ' '.join(command[command.index('get')+1 : command.index('weather')])
                chatbot.VC.speak(f"Fetching {city} weather")
                return get_City_Data(city)
            
            if pos := command.find('of') != -1:
                city = command[pos+2:]
                chatbot.VC.speak(f"Fetching {city} weather")
                return get_City_Data(city)
            
            return "not done"
        

        # If Command contains any of the Reserved Commands
        if any(cmd in command for cmd in chatbot.BasicCmds):
            if pos := command.find('press') != -1:
                # Clean and extract keys
                keys = command[pos+5:].split()
                ag.hotkey(*keys)

            if pos := command.find('open') != -1:
                app_site_name = command[pos+4:]
                chatbot.VC.speak(f"Opening {app_site_name}")
                chatbot.Open(app_site_name)
                
            if pos := command.find('play song') != -1:
                songName = command[pos+9:]
                chatbot.VC.speak(f"Playing song {songName}")
                pwk.playonyt(songName)
            
            if pos := command.find('play') != -1:
                songName = command[pos+4:]
                chatbot.VC.speak(f"Playing {songName}")
                pwk.playonyt(songName)
            
            if pos:= command.find('search') != -1:
                query = command[pos+6:]
                chatbot.VC.speak(f"Searching Google for {query}")
                SearchGoogle(query)

            return "not done"



        #except: print("Error")




    # Chatbot
    def chatbot(self, DA_Name = 'Jarvis'):
        # try:
        # print('Hi, How can I help!')
        chatbot.VC.speak()
        try:
            while True:
                command = chatbot.VC.listen()
                if command == 'stop': break
                while command == "nothing heard":
                    command = chatbot.VC.listen()
                print(f"You: {command}")
                
                command = chatbot.inspect_Speaker_Cmd(command)

                if command != 'not done':
                    intents = chatbot.predict_Class(command, self.words, self.classes)
                    result = chatbot.get_Response(intents, self.data)
                    result = chatbot.refactor_Model_Response(result, command)
                    chatbot.VC.speak(result)
                
        except Exception: return
        # except Exception: return 'Sorry! Something Went Wrong'



chatbot().chatbot()