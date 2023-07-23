This is the code repository for Ghosts in a Shell, seen at ALIFE23.

For this project, I used an HTC Vive headset (none of the controllers or silly floor-stick-camera things), 
Ubuntu 22.04 Jammin Jelly (you can NOT use WSL or WSL2 for this... sorry... I've tried),
and CUDA-capable graphics card NVIDIA RTX 3070.

The idea is first to install CUDA, then create your venv in Python 3.10, then install a bunch of packages.
I made a requirements file to show the exact Python 3.10 packages I installed in my venv. 

In addition, I used this combination:
# cuda 11.7 (on ubuntu 22.04, this worked, but I had to do it twice, install, uninstall, then reinstall: https://gist.github.com/primus852/b6bac167509e6f352efb8a462dcf1854)
# torch 1.13.1, torchaudio 0.13.1, torchvision 0.14.1: pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

Then, use a USB camera and pretend it is the camera and mic from the headset. That is just to ensure all your Python-side software is working correctly.

The final step is to get Linux to recognize the camera/mic on the Vive as a webcam and the video display as a second monitor.
To do that, you'll need to install SteamVR for Linux (https://www.addictivetips.com/ubuntu-linux-tips/steam-vr-on-linux/), 
then make sure you do these steps (https://github.com/ValveSoftware/SteamVR-for-Linux/blob/master/README.md).
You might have to change some NVIDIA driver files manually.

Just so you know, I couldn't actually get SteamVR to open; it kept throwing a 307 error. 
But that's alright; just the process of installing SteamVR installed something critical anyways.
It takes a lot of Linux troubleshooting to work!

Random tip, don't install Chrome. It will break the driver permissions. If you already have Chrome, uninstall it and then unistall SteamVR and then reinstall SteamVR.

Test if the camera/mic works by using "sudo cheese" in the terminal.

If everything works, you should see two windows pop up showing the live feed from your Vive!
Go ahead and drag those windows into the Vive view (one window per eye).

# alife23GHOST
