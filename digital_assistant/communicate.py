import pyttsx3
import speech_recognition as sr





# listener:sr.Recognizer = sr.Recognizer()
# source:sr.Microphone = sr.Microphone()
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)
DA_Name = "Jarvis"

# def  init(self, DA_Name:str = "Jarvis", voiceId:int = 0) -> None:
#     self.DA_Name = DA_Name
#     """Digital Assistant Name -> default: Jarvis"""

#     try: engine.setProperty('voice', voices[voiceId].id)
#     except: engine.setProperty('voice', voices[0].id)



def listen(ackListening:bool = True, language:str = 'en-US', defReturnStmt:str = "nothing heard"):
    """ Listener
        ========
        Uses device microphone to listen to the speaker

        Parameters:
        -----------
        1. ackListening  : bool - If true, acknowledges that listening process has initiated by printing feedback message to console
        2. language : str - language in which input voice is to be recognized by 'recognize_google'
        3. defReturnStmt : str  - default return statement when nothing is heard
        
        Returns:
        --------
        str: if speech recognized, returns text/command listened, else returns defReturnStmt

        Requirements:
        -----------------------
        1. speech_recognition
        2. pyttsx3
    """

    listener = sr.Recognizer()

    #try:
    with sr.Microphone() as source:
        print(f"{DA_Name} :    I AM LISTENING.....")
        audio = listener.listen(source)
        command = listener.recognize_google(audio, language = 'en-US').lower()
        command = ' '.join(command.split())
        return command
    # except: return defReturnStmt




engine  = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(Str = "Hi, how can I help", ackSpeaking:bool = True):
    """ Speak
        =====
        Uses device speaker to provide audio output of the input text

        Parameters:
        -----------
        1. ackListening  : bool - If true, acknowledges that listening process has initiated by printing feedback message to console
        2. language : str - language in which input voice is to be recognized by 'recognize_google'
        3. defReturnStmt : str  - default return statement when nothing is heard
        
        Returns:
        --------
        str: if speech recognized, returns text/command listened, else returns defReturnStmt

        Requirements:
        -----------------------
        1. speech_recognition
        2. pyttsx3
    """

    print(f"{DA_Name} :   ", Str)
    engine.say(Str)
    engine.runAndWait()
    return
