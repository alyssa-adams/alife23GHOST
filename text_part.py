# Python program to translate
# speech to text and text to speech

import speech_recognition as sr
import ecco
import os


class ListenPart:

    def __init__(self):

        # Initialize the recognizer
        self.r = sr.Recognizer()
        self.r.pause_threshold = 0.1  # seconds of non-speaking audio before a phrase is considered complete
        self.r.phrase_threshold = 0.05  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        self.r.non_speaking_duration = 0.05  # seconds of non-speaking audio to keep on both sides of the recording

    def listen(self):

        # wait for a second to let the recognizer
        # adjust the energy threshold based on
        # the surrounding noise level
        print("Adjusting for ambient noise")
        with sr.Microphone() as source2:
            self.r.adjust_for_ambient_noise(source2, duration=5)
        print("Ambient noise adjusted")

        # Loop infinitely for user to speak
        while (1):

            try:

                # use the microphone as source for input.
                with sr.Microphone() as source2:

                    # listens for the user's input
                    audio2 = self.r.listen(source2)

                    # Using google to recognize audio
                    text_to_show = self.r.recognize_google(audio2)
                    text_to_show = text_to_show.lower()

            except sr.RequestError:
                text_to_show = "?????"

            except sr.UnknownValueError:
                text_to_show = "?????"

            f = open("text", "a")
            f.write(str(text_to_show))
            f.close()


class TextAttnPart:

    def __init__(self):

        # load pretrain models
        self.lm = ecco.from_pretrained('gpt2')
        print("GPT2 loaded")

    def text_attn(self):

        if os.path.isfile('text'):

            try:

                f = open("text", "r")
                spoken_words = f.read()
                print(spoken_words)

                output = self.lm.generate(spoken_words, generate=1, do_sample=False, attribution=['ig'])
                output.primary_attributions(attr_method='ig')
                attention_weights = list(output.attribution['ig'][0])

                # TODO: loop through all lengths

                f = open("text_attns", "a")
                f.write(str([spoken_words, attention_weights]))
                f.close()

            except:
                pass
