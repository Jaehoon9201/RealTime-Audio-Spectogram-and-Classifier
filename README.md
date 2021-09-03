# ▶RealTimeAudio-FFT-SPECGRAM-AQ.py
This code is based on the [Reference](https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python).
### [RealTime-Audio-FFT-and-Specgram.py]
1. More optimized. It shows the data acquired during 4 seconds.
2. It saves the audio file obtained during 4 seconds. It also makes a list of the saved files. It could be used for training a audio-classifier.
3. Data type of 'Buffer' is set to 'float'.

### Purpose of this Code 
1. FFT-and-Specgram plotting of Real Time Audio data. 
2. Saving audio data and list of those files.

### Results
The following result shows the result of this code with microphone connected to PC.
Sine wave audio for testing is avalable on this [site](https://www.szynalski.com/tone-generator/)

![image](https://user-images.githubusercontent.com/71545160/131115427-42a692a5-26a9-449a-95f6-bc3a8e57121d.png)

![image](https://user-images.githubusercontent.com/71545160/131115437-8284c60d-74ca-435f-9f7a-5e3d364d4658.png)

![image](https://user-images.githubusercontent.com/71545160/131115481-ddfcd4a7-e98d-412a-be42-ddda25dcb752.png)

![image](https://user-images.githubusercontent.com/71545160/131115514-78c40053-4903-4be2-93a7-543c81b900de.png)


# ▶Audio-DL-Simple-Classifier.py
This code is based on the [Reference](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)

After acquisitioning data with RealTimeAudio-FFT-SPECGRAM-AQ.py,

We can train a model which can classify the real-time audio data coming from serial data.

This code is for the training the classifier.

Data domain is transformed to a frequency domain using a spectogram. It helps the model to be learned easily.

If you want to watch the trained spectogram figure, activate the below code on the file.

```python
# if i % 16 == 0:    
#     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
#     print(data[0].to(device))
#     temp_data = data[0].to(device)
#     temp_data_array = temp_data[0,0,:,:].cpu().data.numpy()

#     plot_spectrogram(temp_data_array, title=None, ylabel='freq_bin', aspect='auto', xmax=None)
```

### Results
![image](https://user-images.githubusercontent.com/71545160/131941710-311b98b4-3029-4825-a6ee-01d20b4829db.png)

# ▶Audio-DL-MelSpecto-Classifier.py
This code is also based on the [Reference](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)

Data domain is transformed to the frequency domain using a Melspectogram.



