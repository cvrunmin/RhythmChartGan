import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import scipy
import scipy.signal

from utils.audio_utils import load_audio


def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g != 0:
        X[X < 0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X


def get_log_freq_mat(freq_bank, freq_res, fc, tc, NumPerOct):
    '''
    get a transform matrix that map a spectrum from linear scale to log scale
    '''
    StartFreq = fc
    StopFreq = tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest - 1, len(freq_bank)), dtype=np.float)
    for i in range(1, Nest - 1):
        l = int(round(central_freq[i - 1] / freq_res))
        r = int(round(central_freq[i + 1] / freq_res) + 1)
        # rounding1
        if l >= r - 1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if freq_bank[j] > central_freq[i - 1] and freq_bank[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (freq_bank[j] - central_freq[i - 1]) / (
                                central_freq[i] - central_freq[i - 1])
                elif freq_bank[j] > central_freq[i] and freq_bank[j] < central_freq[i + 1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - freq_bank[j]) / (
                                central_freq[i + 1] - central_freq[i])
    return freq_band_transformation, central_freq


def get_quef_log_freq_mat(q, fs, fc, tc, NumPerOct):
    '''
    get a transform matrix that map a cepstrum from linear scale to log scale
    '''
    StartFreq = fc
    StopFreq = tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1 / q
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
    for i in range(1, Nest - 1):
        for j in range(int(round(fs / central_freq[i + 1])), int(round(fs / central_freq[i - 1]) + 1)):
            if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (central_freq[i] - central_freq[i - 1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    return freq_band_transformation, central_freq


def get_stft_info(x, freq_res, sample_freq, hop_length):
    t = np.arange(hop_length, np.ceil(len(x) / float(hop_length)) * hop_length, hop_length)
    N = int(sample_freq / float(freq_res))
    f = sample_freq * np.linspace(0, 0.5, np.round(N / 2).astype(int), endpoint=True)

    return f, t, N


def pseudo_stft(x, t, N, window_array):
    window_size = len(window_array)
    Lh = int(np.floor(float(window_size - 1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)

    for icol in range(0, len(t)):
        ti = int(t[icol])
        tau = np.arange(int(-min([round(N / 2.0) - 1, Lh, ti - 1])), \
                        int(min([round(N / 2.0) - 1, Lh, len(x) - ti])))
        indices = np.mod(N + tau, N)
        tfr[indices, icol] = x[ti + tau - 1] * window_array[Lh + tau - 1] / np.linalg.norm(window_array[Lh + tau - 1])
    return abs(scipy.fft.fft(tfr, n=N, axis=0))


def get_CFP(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    '''
    get CFP pack from audio samples
    note that this method receives upper bound frequency as "tc" instead of its reciprocal from original source
    '''
    NumofLayer = np.size(g)
    if NumofLayer != 3:
        raise ValueError('expected to receive 3 gamma values')
    f, t, N = get_stft_info(x, fr, fs, Hop)
    # frq_idx_cap = int(round(N/2))
    HighFreqIdx = int(round(tc / fr) + 1)
    HighQuefIdx = int(round(fs / fc) + 1)
    tc_idx = round(fs / tc)
    fc_idx = round(fc / fr)

    f = f[:HighFreqIdx]
    q = np.arange(HighQuefIdx) / float(fs)

    frq_mapping, central_freq = get_log_freq_mat(f, fr, fc, tc, NumPerOctave)
    qrf_mapping, central_freq = get_quef_log_freq_mat(q, fs, fc, tc, NumPerOctave)

    frame_count = len(t)
    BATCH = 1024  # batch constant: to create CFP batch by batch
    stft_list = []
    gcos_list = []
    gc_list = []
    for idx_range in [range(i, min([i + BATCH, frame_count])) for i in range(0, frame_count, BATCH)]:
        stft = pseudo_stft(x, t[idx_range], N, h)
        stft = np.power(abs(stft), g[0])
        ceps = np.real(np.fft.fft(stft, axis=0)) / np.sqrt(N)
        ceps = nonlinear_func(ceps, g[1], tc_idx)
        tfr = np.real(np.fft.fft(ceps, axis=0)) / np.sqrt(N)
        tfr = nonlinear_func(tfr, g[2], fc_idx)
        stft = stft[:HighFreqIdx, :]
        tfr = tfr[:HighFreqIdx, :]
        ceps = ceps[:HighQuefIdx, :]
        stft = np.dot(frq_mapping, stft)
        tfr = np.dot(frq_mapping, tfr)
        ceps = np.dot(qrf_mapping, ceps)
        stft_list.append(stft)
        gcos_list.append(tfr)
        gc_list.append(ceps)

    stft = np.concatenate(stft_list, axis=1)
    gcos = np.concatenate(gcos_list, axis=1)
    gc = np.concatenate(gc_list, axis=1)

    return stft, gcos, gc, t, central_freq


def feature_extraction(x, fs, Hop=512, Window=2049, StartFreq=80.0, StopFreq=1000.0, NumPerOct=48):
    '''
    cloned from the cfp module of MSnet
    '''
    fr = 2.0  # frequency resolution
    h = scipy.signal.blackmanharris(Window)  # window size
    g = np.array([0.24, 0.6, 1])  # gamma value

    tfrL0, tfrLF, tfrLQ, t, CenFreq = get_CFP(x, fr, fs, Hop, h, StartFreq, StopFreq, g, NumPerOct)
    time = t / fs  # t received is in sample unit, we convert it to time unit here
    return time, CenFreq, tfrL0, tfrLF, tfrLQ


def lognorm(x):
    '''
    cloned from the cfp module of MSnet
    '''
    return np.log(1 + x)


def norm(x):
    '''
    cloned from the cfp module of MSnet
    '''
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def cfp_process(fpath, sr=None, hop=256, fps=None, model_type='vocal'):
    """
    cloned from the cfp module of MSnet
    """
    print('CFP process in ' + str(fpath) + ' ... (It may take some times)')
    y, sr = load_audio(fpath, sr=sr)
    if hop is None and fps is not None:
        hop = sr // fps
    if 'vocal' in model_type:
        time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, StartFreq=31.0, StopFreq=1250.0,
                                                                NumPerOct=60)
    if 'melody' in model_type:
        time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, StartFreq=20.0, StopFreq=2048.0,
                                                                NumPerOct=60)
    tfrL0 = norm(lognorm(tfrL0))[np.newaxis, :, :]
    tfrLF = norm(lognorm(tfrLF))[np.newaxis, :, :]
    tfrLQ = norm(lognorm(tfrLQ))[np.newaxis, :, :]
    W = np.concatenate((tfrL0, tfrLF, tfrLQ), axis=0)
    print('Done!')
    print('Data shape: ' + str(W.shape))
    return W, CenFreq, time
