def WavSpectrum(audioName, show_plots):
    from scipy.io import wavfile
    import numpy as np

    fs, audiodata = wavfile.read(audioName)
    
    if audiodata.ndim == 2:
        audiodata = StereoToMono(audiodata)
    N = audiodata.shape[0]
    print("Liczba sampli: ", N)
    sec = N/float(fs)
    print("Czas trwania: {}s".format(sec))
    T = 1.0/fs
    t = np.arange(0, sec, T)

    from scipy.fftpack import fft, fftfreq
    FFT = np.abs(fft(audiodata))
    FFT = FFT[range(N//2)]
    freqs = fftfreq(audiodata.size, t[1] - t[0])
    freqs = freqs[range(N//2)]

    if bool(show_plots):
        import matplotlib.pyplot as plt
        plt.plot(t, audiodata, "g")
        plt.xlabel("Czas[s]")
        plt.ylabel("Amplituda")
        plt.title("Audio waveform w czasie", size = 16)

        plt.figure()
        plt.plot(freqs, abs(FFT), "r")
        plt.xlabel('Częstotliwość [Hz]') 
        plt.ylabel('Wzmocnienie (dB)')
        plt.title('Widmo',size=16)

        #print(Audiodata)
        from scipy import signal
        N = 512

        f, t, S = signal.spectrogram(audiodata, fs, window = signal.blackman(N), nfft = N)
        plt.figure()
        plt.pcolormesh(t, f, 10*np.log10(S))

        plt.ylabel('Częstotliwość [Hz]')
        plt.xlabel('Czas [seg]')
        plt.title('Spektrogram',size=16)

        plt.show()

    return FFT

def legacySpectrum(AudioName, show_plots):  
    from scipy.io import wavfile
    import numpy as np

    fs, Audiodata = wavfile.read(AudioName)

    if (Audiodata.ndim > 1):
        data = StereoToMono(Audiodata)
    else:
        data = Audiodata
    
    from scipy.fftpack import fft
    #print(data)
    n = len(data)
    AudioFreq = fft(data)
    AudioFreq = AudioFreq[0:int(np.ceil((n+1)/2.0))]
    MagFreq = np.abs(AudioFreq)
    MagFreq = MagFreq / float(n)
    MagFreq = MagFreq**2
    if n % 2 > 0: # ffte odd 
        MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
    else:# fft even
        MagFreq[1:len(MagFreq) -1] = MagFreq[1:len(MagFreq) - 1] * 2 

    if bool(show_plots):
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.title("Audio signal in time", size = 16)

        plt.figure()
        freqAxis = np.arange(0, int(np.ceil((n+1)/2.0)), 1.0) * (fs/n)
        plt.xscale('linear')
        plt.plot(freqAxis/1000.0, 10*np.log10(MagFreq))
        plt.xlabel('Frequency (kHz)') 
        plt.ylabel('Power spectrum (dB)')
        plt.title('Spectrum',size=16)

        #print(Audiodata)
        from scipy import signal
        N = 512

        f, t, S = signal.spectrogram(data, fs, window = signal.blackman(N), nfft = N)
        plt.figure()
        plt.pcolormesh(t, f, 10*np.log10(S))

        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [seg]')
        plt.title('Spectrogram',size=16)

        plt.show()

    return AudioFreq, MagFreq

def StereoToMono(data):
    import numpy as np
    result = []
    for x in data:
        suma = x[0] + x[1]
        result.append(suma/2)
    np.seterr(all='warn')
    result = np.array(result, dtype = "int64")
    result = result.T
    return result