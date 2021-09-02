# coding=<utf-8>


import pandas as pd
from pathlib import Path
import math, random
import torch
import torchaudio
import numpy as np
from torchaudio import transforms
from IPython.display import Audio

from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torch.utils.data import random_split


import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn

import sklearn
import librosa
import librosa.display

from matplotlib import pyplot as plt
import tensorflow as tf




#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Read csv data file
train_csvdata_file = 'data/train_class.csv'
train_data_path = 'data/train'   
train_df = pd.read_csv(train_csvdata_file)

test_csvdata_file = 'data/test_class.csv'
test_data_path = 'data/test'   
test_df = pd.read_csv(test_csvdata_file)

train_df = train_df.values[1:, :]
train_df = pd.DataFrame(train_df, columns = ['cnt', 'classID', 'relative_path'])
train_df = train_df.loc[:, ['relative_path', 'classID']]

test_df = test_df.values[1:, :]
test_df = pd.DataFrame(test_df, columns = ['cnt', 'classID', 'relative_path'])
test_df = test_df.loc[:, ['relative_path', 'classID']]


#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

  # ----------------------------
  # Convert the given audio to the desired number of channels
  # ----------------------------
  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))


 
  # ----------------------------
  # Generate a Spectrogram
  # ----------------------------
  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

# ----------------------------
# train_Sound Dataset
# ----------------------------
class train_SoundData(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        audio_file = self.data_path + '/' + self.df.loc[idx, 'relative_path']
        class_id = self.df.loc[idx, 'classID']
        aud_data = AudioUtil.open(audio_file)
        rechanneled = AudioUtil.rechannel(aud_data, self.channel)
        spectrogram = AudioUtil.spectro_gram(rechanneled, n_mels=64, n_fft=1024, hop_len=None)
        

        return spectrogram, class_id

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ----------------------------
# test_Sound Dataset
# ----------------------------
class test_SoundData(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        audio_file = self.data_path + '/' + self.df.loc[idx, 'relative_path']
        class_id = self.df.loc[idx, 'classID']
        aud_data = AudioUtil.open(audio_file)
        rechanneled = AudioUtil.rechannel(aud_data, self.channel)
        spectrogram = AudioUtil.spectro_gram(rechanneled, n_mels=64, n_fft=1024, hop_len=None)
        

        return spectrogram, class_id

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■




# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x



#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # ■■■■■■■■■■
            # for a monitoring
            # ■■■■■■■■■■
            # if i % 16 == 0:    
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            #     print(data[0].to(device))
            #     temp_data = data[0].to(device)
            #     temp_data_array = temp_data[0,0,:,:].cpu().data.numpy()

            #     print(temp_data[0,0,:,:])  
            #     print(temp_data[0,0,:,:].cpu().data.numpy()) 
            #     print(temp_data_array.shape)
            #     print(temp_data_array[:,-1])

            #     plt.plot(temp_data_array[:,-1])
            #     plt.show()
            #     plt.imshow(temp_data_array, cmap='viridis', aspect='auto') # cmap='viridis', 'viridis_r', 'inferno', 'inferno_r', 'plasma', plt.cm.Blues, plt.cm.Blues_r, 'BrBG', 'BrBG_r'
            #     plt.show()
            # ■■■■■■■■■■
            # ■■■■■■■■■■
            # ■■■■■■■■■■
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('Finished Training')
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

# ----------------------------
# Testing
# ----------------------------
def testing(model, test_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')


#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■  Main  ■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device


train_ds = train_SoundData(train_df, train_data_path)
test_ds = test_SoundData(test_df, test_data_path)



# Create training and validation data loaders
print('data loading- start')
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)
print('data loading- finished')


num_epochs = 40  
print('Starting Training')
training(myModel, train_dl, num_epochs)
testing(myModel, test_dl)

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■