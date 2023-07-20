# Alyssa Adams, Nicholas Guttenberg
# Cross Labs in Kyoto Japan
# 2023, for ALIFE23

import os
import sys
import ast
import cv2
import math
import time
import numpy as np
from PIL import Image
from multiprocessing import Process, set_start_method

# for the video
import clip
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
# cuda 11.7 (on ubuntu 22.04, this worked, but I had to do it twice, install, uninstall, then reinstall: https://gist.github.com/primus852/b6bac167509e6f352efb8a462dcf1854)
# torch 1.13.1, torchaudio 0.13.1, torchvision 0.14.1: pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# for the text
from text_part import ListenPart, TextAttnPart

# supress warnings
#os.close(sys.stderr.fileno())


def video_part():

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
        attn_pat, attn_weights = self.attn(z, z, z, need_weights=True, average_attn_weights=True, attn_mask=attn_mask) # average_attn_weights=True,
        attn_pat = attn_pat[0]

        x = x + attn_pat
        x = x + self.mlp(self.ln_2(x))

        return x, attn_weights

    def attn_mask(frame, lastw, layer):

        with torch.no_grad():

            with torch.autocast("cuda"):
                frame = Image.fromarray(frame[:456, 612//2-228:612//2+228, :]) # 456x456 slice
                frame = frame.resize((336, 336), Image.NEAREST)

                z = preprocess(frame).unsqueeze(0).cuda()
                z, attn = model.visual.vision_attn_forward(z)

                # attn = torch.cat(attn,0).mean(0)[0,1:]
                attn = torch.cat(attn, 0)
                attn = attn[layer, 0, 1:] # Visualize the last attention layer
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

    # variables for the video stream
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.visual.vision_attn_forward = vision_attn_forward.__get__(model.visual) # model.visual -> self
    model.visual.transformer.transformer_attn_forward = transformer_attn_forward.__get__(model.visual.transformer) # model.visual.transformer -> self
    for layer in model.visual.transformer.resblocks:
        layer.resblock_attn_forward = resblock_attn_forward.__get__(layer)
    vid = cv2.VideoCapture(-1)
    lastw = np.zeros((336, 336, 1))  # get the right frame size automatically

    # set initial values for video loop
    display_words = False
    display_weights = False
    layer = 0

    while True:

        # loop over video frames and cycle through attention layers
        ret, frame = vid.read()
        frame, lastw = attn_mask(frame, lastw, layer//2)
        layer = (layer+1) % 47

        # look for spoken words, if yes then turn on display_words variable
        if os.path.isfile('text_attns'):

            f = open("text_attns", "r")
            contents = f.read()
            contents = ast.literal_eval(contents)
            display_words = contents[0]
            display_weights = contents[1]

            # leave the text up for a few frames
            t0 = time.time()

            # todo
            # or until a new phrase has been spoken

        # if display_words is on, display the words on the existing frame
        if display_words:

            coordinates = (100, 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (161, 164, 255)
            thickness = 1
            frame = cv2.putText(frame, display_words, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

            """
            for i, word in enumerate(text_to_show.split()):

                space_between_words = 20
                padding = 50

                coordinates = (padding, padding + space_between_words * i)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = attention_weights[i] * len(text_to_show.split())
                color = (161, 164, 255)
                thickness = 1
                frame = cv2.putText(frame, word, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            """

        # show the image
        scale = 1
        frame = cv2.resize(frame, (336 * scale, 336 * scale))
        cv2.imshow('frame1', frame)
        cv2.imshow('frame2', frame)

        # if display-words is on, check how long they've been up
        # 1 second per phrase, + 0.5 second per word after that
        # turn display_words variable off again
        # delete the text and text_attn files
        if display_words:

            tf = time.time()
            dt = tf - t0
            if dt > 1 + len(display_words.split()) / 2:

                display_words = False
                os.remove("text")
                os.remove("text_attns")

        # quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
    os.remove("text")
    os.remove("text_attns")
    quit()


def listen_part():

    listener = ListenPart()
    while True:
        if os.path.isfile('text'):
            continue
        else:
            listener.listen()


def text_attender_part():

    textattender = TextAttnPart()
    while True:
        if os.path.isfile('text'):
            continue
        else:
            textattender.text_attn()


if __name__ == '__main__':

    set_start_method("spawn")

    p1 = Process(target=video_part)
    p2 = Process(target=listen_part)
    p3 = Process(target=text_attender_part)

    p1.start()
    p2.start()
    p3.start()

    #video_part()

    quit()
