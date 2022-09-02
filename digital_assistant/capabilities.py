
##################################### Method to get City Data using OpenWeathermap API #####################################

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
        city data using OpenWeathermap API
    """
	
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






##################################### Method to get Current Date-Time #####################################

from datetime import datetime


def get_DateTime(data:str = 'date-time'):
    """Get Date Time
        ================

        Parameters:
        -----------
        1. data:str - Values('date-time', 'accurate-date-time', 'time', 'accurate-time') returns data according to the choice
        
        Returns:
        --------
        date-time data using Date-Time API
    """
    
    months = ('January', 'February', 'March', 'April')
    cDT = datetime.now()                                                # current date-time
    desiredData = {
        'date-time': f"{cDT:%B %d, 20%y, %H:%M}",
        'time': f"{cDT:%H:%M}",
        'accurate-time': f"{cDT:%H : %M : %S.%f}",
        'accurate-date-time': f"{cDT:%B %d, 20%y, %H:%M:%S.%f}"
    }

    return desiredData.get(data.strip(), 'unknown')





##################################### Method to Scrap Dictionary #####################################

from PyDictionary import PyDictionary

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






##################################### Method to Surf Google #####################################

# import google_images_search
import googlesearch as google


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