import speech_recognition as sr
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the voice and rate
engine.setProperty('voice', 'english')
engine.setProperty('rate', 150)

# Initialize the recognizer
r = sr.Recognizer()

# Ask the user for their name
engine.say("What is your name?")
engine.runAndWait()

# Use the microphone as the audio source
with sr.Microphone() as source:
    # Listen for the user's response
    audio = r.listen(source)

# Try to recognize the user's response
try:
    # Store the user's response in a variable
    user_name = r.recognize_google(audio)
    print("Your name is: " + user_name)
    # Speak the result
    engine.say("Your name is " + user_name)
    engine.runAndWait()
except sr.UnknownValueError:
    engine.say("I'm sorry, I didn't understand what you said.")
    engine.runAndWait()
except sr.RequestError as e:
    engine.say("I'm sorry, my speech service is currently down.")
    engine.runAndWait()
