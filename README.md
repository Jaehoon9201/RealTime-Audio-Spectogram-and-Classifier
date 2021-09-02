# RealTimeAudio-FFT-SPECGRAM-AQ.py
It's a newly updated code from the below file(RealTime-Audio-FFT-and-Specgram).
It's also based on the [Reference](https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python).

### what is different with [RealTime-Audio-FFT-and-Specgram.py]
:bulb: More optimized. It shows the data acquired during 2 seconds.

:bulb: It saves the audio file obtained during 2 seconds. It also makes a list of the saved files.

:bulb: Data type of 'Buffer' is changed to 'float'.

### Purpose of this Code 
FFT-and-Specgram plotting of Real Time Audio data

### RealTime-Audio-FFT-and-Specgram.py
The following result shows the result of this code with microphone connected to PC.
Sine wave audio for testing is avalable on this [site](https://www.szynalski.com/tone-generator/)

### Results
![image](https://user-images.githubusercontent.com/71545160/131115427-42a692a5-26a9-449a-95f6-bc3a8e57121d.png)

![image](https://user-images.githubusercontent.com/71545160/131115437-8284c60d-74ca-435f-9f7a-5e3d364d4658.png)

![image](https://user-images.githubusercontent.com/71545160/131115481-ddfcd4a7-e98d-412a-be42-ddda25dcb752.png)

![image](https://user-images.githubusercontent.com/71545160/131115514-78c40053-4903-4be2-93a7-543c81b900de.png)


# Audio-DL-Simple-Classifier-NVH.py
After acquisitioning data with RealTimeAudio-FFT-SPECGRAM-AQ.py,

We can train a model which can classify the real-time audio data coming from serial data.

This code is for the training the classifier.

# RealTime-Audio-FFT-and-Specgram.py
Old version of the [RealTimeAudio-FFT-SPECGRAM-AQ.py]
