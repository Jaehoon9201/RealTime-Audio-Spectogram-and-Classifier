# Reference : https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python
# Based on the reference code, Histogram is added by Jaehoon Shim, 2021-07-31

# Reference code is revised to float type data and added plt.pause function

import pyaudio
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import time
from tkinter import TclError
import librosa, librosa.display
from matplotlib.mlab import window_hanning,specgram
from matplotlib.colors import LogNorm

# constants
CHUNK = 1024 * 16            # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second

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
ax1.set_ylim(0, 1023)
ax1.set_xlim(0, 2 * CHUNK)
plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 512, 1023])

# format spectrum axes
ax2.set_xlim(20, RATE / 2)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()
ctr = 0

while True:

    # binary data
    data = stream.read(CHUNK)
    data_int = struct.unpack('h' * CHUNK, data)
    # create np array and offset
    data_np = np.array(list(data_int), dtype='float') + 512
    line.set_ydata(data_np)
    # compute FFT and update line
    yf = fft(data_int)
    line_fft.set_ydata(np.abs(yf[0:CHUNK]) / (128 * CHUNK))


    # SPECTGRAM
    arr2D, freqs, bins = specgram(data_np, window=window_hanning, Fs=RATE, NFFT=1024, noverlap=512)
    extent = (bins[0], bins[-1] * 32, freqs[-1], freqs[0])
    im = plt.imshow(arr2D, aspect='auto', extent=extent, interpolation="none",  norm=LogNorm(vmin=.01, vmax=1))
    arr2D, freqs, bins = specgram(data_np, window=window_hanning, Fs=RATE, NFFT=1024, noverlap=512)
    im_data = im.get_array()



    # update figure canvas
    try:
        # DISPLAY REAL TIME FFT
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1


        # DISPLAY SPECTGRAM
        if (ctr % 20 == 0):
            if ctr < 64:
                im_data = np.hstack((im_data, arr2D))
                im.set_array(im_data)
            else:
                keep_block = arr2D.shape[1] * (64 - 1)
                im_data = np.delete(im_data, np.s_[:-keep_block], 1)
                im_data = np.hstack((im_data, arr2D))
                im.set_array(im_data)
            fig2.canvas.draw()
        ctr += 1


        # im_data = np.hstack((im_data, arr2D))
        # im.set_array(im_data)
        # keep_block = arr2D.shape[1] * (16 - 1)
        # im_data = np.delete(im_data, np.s_[:-keep_block], 1)
        # im_data = np.hstack((im_data, arr2D))
        # im.set_array(im_data)
        # fig2.canvas.draw()


        plt.pause(0.0000001)


    except TclError:

        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)

        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break