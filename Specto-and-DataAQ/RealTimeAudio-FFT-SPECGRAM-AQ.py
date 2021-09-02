import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import IPython.display as ipd
import soundfile as sf

import pyaudio
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import time
from tkinter import TclError
import librosa, librosa.display
from matplotlib.mlab import window_hanning, specgram
from matplotlib.colors import LogNorm

import csv


# constants
DATA_AQ_TIME = 2
CHUNK = 44100*DATA_AQ_TIME  # samples per frame
FORMAT = pyaudio.paFloat32   # audio format (bytes per sample?)
CHANNELS = 1  # single channel for microphone
RATE = 44100  # samples per second
ClassID = 0  # 0 : Normal
# create matplotlib figure and axes
fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 4))
fig2, ax3 = plt.subplots(1, figsize=(6, 6))
ax3.set_title('Spectrogram (dB)')
ax3.set_xlabel('Time')
ax3.set_ylabel('Frequency')

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# variable for plotting
x = np.arange(0, 2 * CHUNK, 2)  # samples (waveform)
xf = np.linspace(0, RATE, CHUNK)  # frequencies (spectrum)

# create a line object with random data
line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=2)

# create semilogx line for spectrum
line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), '-', lw=2)

# format waveform axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-0.4, 0.4)
ax1.set_xlim(0, 2 * CHUNK)
plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-0.4, 0, 0.4])

# format spectrum axes
ax2.set_xlim(20, RATE / 2)
ax2.set_ylim(-0.001, 0.02)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()
ctr = 0
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■ Saving AudioFiles with plotting the FFT and SPECTOGRAM■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

# Saving Class IDs on the CSV file
f = open("acquired_data/AudioName-ClassID.csv", "w", encoding="UTF-8")
f.write('Count' + ',' +'AudioName' + ',' + 'Class ID' + '\n')
while True:

    # binary data
    data = stream.read(CHUNK)
    data_int = struct.unpack('f' * CHUNK, data)
    # create np array and offset
    #data_np = np.array(list(data_int), dtype='float') + 512
    data_np = np.array(list(data_int), dtype='float64')
    print(data_np)
    print(data_np.shape)
    line.set_ydata(data_np)

    # compute FFT and update line
    yf = fft(data_int)
    line_fft.set_ydata(np.abs(yf[0:CHUNK]) / CHUNK)

    # SPECTGRAM
    arr2D, freqs, bins = specgram(data_np, window=window_hanning, Fs=RATE, NFFT=1024, noverlap=512)
    print('bins')
    print(bins)
    print(bins[0])
    print(bins[-1])
    #extent = (bins[0], bins[-1] * 32, freqs[-1], freqs[0])
    extent = (bins[0], bins[-1], freqs[-1], freqs[0])
    print(arr2D)
    print(arr2D.shape)
    im = plt.imshow(arr2D, aspect='auto', extent=extent, interpolation="none", norm=LogNorm(vmin=10**-16, vmax=10**-2))
    arr2D, freqs, bins = specgram(data_np, window=window_hanning, Fs=RATE, NFFT=1024, noverlap=512)
    im_data = im.get_array()

    # SAVE DATA
    ipd.Audio(data_np, rate=RATE)  # load a NumPy array
    sf.write('acquired_data/Data Num %d ClassID %d.wav' % (ctr, ClassID), data_np, RATE, 'PCM_24')
    f = open("acquired_data/AudioName-ClassID.csv", "a", encoding="UTF-8", newline='')  # not a "w" mode !
    f.write('%d' %(ctr)+','+ 'Data Num %d ClassID %d.wav' % (ctr, ClassID) + ',' + ' %d' % (ClassID)+ '\n')
    f.close()

    try:
        # DISPLAY REAL TIME FFT
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1
        im.set_array(im_data)
        fig2.canvas.draw()
        ctr += 1
        plt.pause(0.4)


    except TclError:
        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)
        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break

