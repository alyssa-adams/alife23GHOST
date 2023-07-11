import math

import cv2
import numpy as np
from PIL import Image

# for the images
import clip
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

# for the text
import speech_recognition as sr
import pyttsx3


# for the images
def vision_attn_forward(self, x):

    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
         x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x, attn = self.transformer.transformer_attn_forward(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_post(x[:, 0, :])

    if self.proj is not None:
        x = x @ self.proj

    return x, attn


def transformer_attn_forward(self, x):

    z = x
    attns = []

    for layer in self.resblocks:
        z, a = layer.resblock_attn_forward(z)
        attns.append(a)

    return z, attns


def resblock_attn_forward(self, x):

    attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

    z = self.ln_1(x)
    attn_pat, attn_weights = self.attn(z, z, z, need_weights=True, average_attn_weights=True, attn_mask=attn_mask)
    attn_pat = attn_pat[0]

    x = x + attn_pat
    x = x + self.mlp(self.ln_2(x))

    return x, attn_weights


def attn_mask(frame, lastw):

    with torch.no_grad():

        with torch.autocast("cuda"):
            
            frame = Image.fromarray(frame[:480, 80:80 + 480, :])
            frame = frame.resize((336, 336), Image.NEAREST)

            z = preprocess(frame).unsqueeze(0).cuda()
            z, attn = model.visual.vision_attn_forward(z)

            # attn = torch.cat(attn,0).mean(0)[0,1:]
            attn = torch.cat(attn, 0)[-1, 0, 1:]
            GRID = int(math.sqrt(attn.shape[0]))
            attn = attn.view(1, 1, GRID, GRID)
            attn = F.upsample_bilinear(attn.view((1, 1, GRID, GRID)), scale_factor=frame.width // GRID)
            attn = attn.cpu().detach().numpy()[0, 0]

            im2 = np.array(frame).astype(np.float32)
            w = np.clip((attn / 0.0025) ** 6, 0, 1)[:, :, np.newaxis]
            w = 0.25 * w + 0.75 * lastw
            im2 = w * im2 + np.mean(im2, axis=(0, 1), keepdims=True) * (1 - w)
            im2 = np.clip(im2, 0, 255).astype(np.uint8)

            return im2, w


# for the text
# initialize the speech recognizer
r = sr.Recognizer()
r.pause_threshold = 0.5  # seconds of non-speaking audio before a phrase is considered complete
r.phrase_threshold = 0.2  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
r.non_speaking_duration = 0.1  # seconds of non-speaking audio to keep on both sides of the recording


if __name__ == '__main__':

    device = "cuda"

    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.visual.vision_attn_forward = vision_attn_forward.__get__(model.visual)
    model.visual.transformer.transformer_attn_forward = transformer_attn_forward.__get__(model.visual.transformer)

    for layer in model.visual.transformer.resblocks:
        layer.resblock_attn_forward = resblock_attn_forward.__get__(layer)

    vid = cv2.VideoCapture(0)
    lastw = np.zeros((336, 336, 1))  # get the right frame size automatically

    text_to_show = ''

    while True:

        # listen to any speech
        try:  # todo: speed this part up. Frames refresh at rate, but this sometimes adds new text. If new text, then add. Two while loops with multiprocessing?

            # use the microphone as source for input.
            with sr.Microphone() as source2:

                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2)

                # listens for the user's input
                audio2 = r.listen(source2)

                # Using google to recognize audio
                text_to_show = r.recognize_google(audio2)
                text_to_show = text_to_show.lower()

        except sr.RequestError as e:
            text_to_show = "?????"

        except sr.UnknownValueError:
            text_to_show = "?????"

        # take the camera image and add the mask
        ret, frame = vid.read()
        frame, lastw = attn_mask(frame, lastw)

        # add the text to the frame
        coordinates = (100, 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 255)
        thickness = 2
        frame = cv2.putText(frame, text_to_show, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

        # show the image
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
    quit()
