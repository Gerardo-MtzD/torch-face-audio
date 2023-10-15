import pyaudio
import io
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import queue
from matplotlib.animation import FuncAnimation
import sys
import argparse
import cv2


from utils.audio_parser import audio_parser

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "../file.wav"
ARGS = audio_parser()
q = queue.Queue()
mapping = [c - 1 for c in ARGS.channels]  # Channel numbers start with 1
if ARGS.samplerate is None:
    device_info = sd.query_devices(ARGS.device, 'input')
    ARGS.samplerate = device_info['default_samplerate']
length = int(ARGS.window * ARGS.samplerate / (1000 * ARGS.downsample))
plotdata = np.zeros((length, len(ARGS.channels)))


class Audio:
    def __init__(self, tk: bool = False):
        self.FORMAT = pyaudio.paInt16
        self.tk = tk
        self.CHANNELS = 2
        self.RATE = 44100
        self.CHUNK = 1024
        self.RECORD_SECONDS = 2
        self.WAVE_OUTPUT_FILENAME = "../file.wav"
        self.ARGS = audio_parser()
        print(self.ARGS)
        self.q = queue.Queue()
        self.mapping = [c - 1 for c in self.ARGS.channels]  # Channel numbers start with 1
        if self.ARGS.samplerate is None:
            self.device_info = sd.query_devices(self.ARGS.device, 'input')
            self.ARGS.samplerate = device_info['default_samplerate']
        self.length = int(self.ARGS.window * self.ARGS.samplerate / (1000 * self.ARGS.downsample))
        self.plotdata = np.zeros((length, len(self.ARGS.channels)))
        self.lines = []

    def stream_audio(self):
        fig, ax = plt.subplots()
        self.lines = ax.plot(self.plotdata)
        if len(self.ARGS.channels) > 1:
            ax.legend(['channel {}'.format(c) for c in self.ARGS.channels],
                      loc='lower left', ncol=len(self.ARGS.channels))
        ax.axis((0, len(self.plotdata), -1, 1))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                       right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)
        stream = sd.InputStream(
            device=self.ARGS.device,
            channels=max(self.ARGS.channels),
            samplerate=self.ARGS.samplerate, callback=self.audio_callback)
        if self.tk:
            ani = FuncAnimation(fig, self.tk_update_graph, interval=self.ARGS.interval, blit=True)
        else:
            ani = FuncAnimation(fig, self.update_graph, interval=self.ARGS.interval, blit=True)
            with stream:
                plt.show()

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        self.q.put(indata[::self.ARGS.downsample, self.mapping])

    def update_graph(self, _):
        while True:
            try:
                data = self.q.get_nowait()
                # print(data)
                # print(np.average(data))
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data

        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, column])
        return self.lines

    def tk_update_graph(self, _):
        data = None
        try:
            data = self.q.get_nowait()
            # print(data)
            print(np.average(data))
        except queue.Empty:
            pass
        if data:
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data

        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, column])
        return self.lines
