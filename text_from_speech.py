# Python program to translate
# speech to text and text to speech

import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()
r.pause_threshold = 0.5  # seconds of non-speaking audio before a phrase is considered complete
r.phrase_threshold = 0.2  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
r.non_speaking_duration = 0.1  # seconds of non-speaking audio to keep on both sides of the recording

# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Loop infinitely for user to
# speak

while (1):

    # Exception handling to handle
    # exceptions at the runtime
    try:

        # use the microphone as source for input.
        with sr.Microphone() as source2:

            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2)

            # listens for the user's input
            audio2 = r.listen(source2)

            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            #MyText = MyText.lower()

            print(MyText)

            # say it back
            # SpeakText(MyText)

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("?????")
