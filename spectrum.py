import string
from matplotlib.pyplot import show
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
"""
    This script generates Audio Waveform Spectrum and Spectrogram
    from a 16bit wav mono or stereo file
"""
def wav_spectrum(audio_name: string, show_plots: bool):
    """
        This method generates a list of values for the spetrum
        using fft function and shows plots based on those values
    """
    fs, audiodata = wavfile.read(audio_name)
    if audiodata.ndim == 2:
        audiodata = StereoToMono(audiodata)
    N = audiodata.shape[0]
    print("Liczba sampli: ", N)
    sec = N/float(fs)
    print("Czas trwania:   {}s".format(sec))
    T = 1.0/fs
    t = np.arange(0, sec, T)
    FFT = np.abs(fft(audiodata))
    FFT = FFT[range(N//2)]
    freqs = fftfreq(audiodata.size, t[1] - t[0])
    freqs = freqs[range(N//2)]
    if show_plots:
        create_plots(audiodata, fs, t, FFT, freqs)
    #return audiodata, fs, FFT, freqs, show_plots

def create_plots(audiodata, fs, t, FFT, freqs):
    plt.plot(t, audiodata, "g")
    plt.xlabel("Czas[s]")
    plt.ylabel("Amplituda")
    plt.title("Audio waveform w czasie", size = 16)
    plt.figure()
    plt.plot(freqs, abs(FFT), "r")
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Wzmocnienie (dB)')
    plt.title('Widmo',size=16)
    from scipy import signal
    N = 512
    f, t, S = signal.spectrogram(audiodata, fs, window = signal.blackman(N), nfft = N)
    plt.figure()
    plt.pcolormesh(t, f, 10*np.log10(S))
    plt.ylabel('Częstotliwość [Hz]')
    plt.xlabel('Czas [seg]')
    plt.title('Spektrogram',size=16)
    plt.show()

def StereoToMono(data):
    result = []
    for x in data:
        suma = x[0] + x[1]
        result.append(suma/2)
    np.seterr(all='warn')
    result = np.array(result, dtype = "int64")
    result = result.T
    return result