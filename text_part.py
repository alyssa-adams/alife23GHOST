# Python program to translate
# speech to text and text to speech

import speech_recognition as sr
import ecco


class TextPart:

    def __init__(self):

        # Initialize the recognizer
        self.r = sr.Recognizer()
        self.r.pause_threshold = 0.5  # seconds of non-speaking audio before a phrase is considered complete
        self.r.phrase_threshold = 0.2  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        self.r.non_speaking_duration = 0.1  # seconds of non-speaking audio to keep on both sides of the recording

        # load pretrain models
        self.lm = ecco.from_pretrained('gpt2')
        print("GPT loaded")

    def listen(self):

        # Loop infinitely for user to speak
        while (1):

            try:

                # use the microphone as source for input.
                with sr.Microphone() as source2:

                    # wait for a second to let the recognizer
                    # adjust the energy threshold based on
                    # the surrounding noise level
                    self.r.adjust_for_ambient_noise(source2)

                    # listens for the user's input
                    audio2 = self.r.listen(source2)

                    # Using google to recognize audio
                    text_to_show = self.r.recognize_google(audio2)
                    text_to_show = text_to_show.lower()

            except sr.RequestError as e:
                text_to_show = "?????"

            except sr.UnknownValueError:
                text_to_show = "?????"

            output = self.lm.generate(text_to_show, generate=1, do_sample=False, attribution=['ig'])
            output.primary_attributions(attr_method='ig')
            attention_weights = list(output.attribution['ig'][0])

            f = open("text", "a")
            f.write(str([text_to_show, attention_weights]))
            f.close()
