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
from text_part import TextPart

# supress warnings
os.close(sys.stderr.fileno())


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

    # variables for the video stream
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.visual.vision_attn_forward = vision_attn_forward.__get__(model.visual)
    model.visual.transformer.transformer_attn_forward = transformer_attn_forward.__get__(model.visual.transformer)
    for layer in model.visual.transformer.resblocks:
        layer.resblock_attn_forward = resblock_attn_forward.__get__(layer)
    vid = cv2.VideoCapture(0)
    lastw = np.zeros((336, 336, 1))  # get the right frame size automatically

    # TODO: frame size for headset
    # TODO: Automatic font spacing
    # TODO: Font color based on average frame color (filler color)
    # TODO: Font lightness/darkness based on weight

    # TODO: BONUS! Fade out text, better font
    # TODO: BONUS! Add in ghost trails

    text_up = False

    while True:

        ret, frame = vid.read()
        frame, lastw = attn_mask(frame, lastw)

        # add in the text
        if os.path.isfile('text'):

            text_up = True

            f = open("text", "r")
            contents = f.read()
            contents = ast.literal_eval(contents)
            text_to_show = contents[0]
            attention_weights = contents[1]
            os.remove("text")

            # leave the text up for a few frames
            t0 = time.time()

        if text_up:

            for i, word in enumerate(text_to_show.split()):

                space_between_words = 20
                padding = 50

                coordinates = (padding, padding + space_between_words * i)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = attention_weights[i] * len(text_to_show.split())
                color = (161, 164, 255)
                thickness = 1
                frame = cv2.putText(frame, word, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

        # show the image
        cv2.imshow('frame', frame)

        # leave the text up for 5 seconds
        if text_up:
            tf = time.time()
            dt = tf - t0
            if dt > len(text_to_show.split())/2:
                text_up = False

        # quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
    quit()


def text_part():

    listener = TextPart()
    while True:
        listener.listen()


if __name__ == '__main__':

    set_start_method("spawn")
    p1 = Process(target=video_part)
    p2 = Process(target=text_part)
    p1.start()
    p2.start()

    quit()
