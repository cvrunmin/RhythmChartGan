import soundfile
import numpy as np
import scipy
from typing import Union
from pathlib import Path


def load_audio(filepath: str, sr: int=None, mono=True, dtype=float):
    """
    Load audio file into ndarray of audio sample. Note that *.mp3 file is not supported.
    :param filepath: file path of the audio
    :param sr: target sampling rate of the result. default None which return the sample as is.
    :param mono: if set to True (default), the audio sample will be convert to mono first by averaging sample w.r.t. channels
    :param dtype: sample data type. default float
    :return: (audio sample, sampling rate of the sample)
    """

    if '.mp3' in filepath:
        raise IOError('mp3 file is not supported')
    else:
        x, fs = soundfile.read(filepath)

    if mono and len(x.shape)>1:
        x = np.mean(x, axis = 1)
    if sr:
        x = scipy.signal.resample_poly(x, sr, fs)
        fs = sr 
    x = x.astype(dtype)

    return x, fs


def get_mel_spectrogram(audio_sample: str, audio_sr:int=44100, frame_length:int=1024, *, frame_step:int=None, fps=100):
    import librosa
    import numpy as np

    f_min, f_max, num_mel_bins = 27.5, 16000.0, 80

    if frame_step is None:
        frame_step = audio_sr // fps

    mel_spec = librosa.feature.melspectrogram(y=audio_sample, sr=audio_sr, n_fft=frame_length, hop_length=frame_step,
        n_mels=num_mel_bins, fmin=f_min, fmax=f_max)
    return np.log(mel_spec+1e-16).astype(float)


def get_lmel80(p: Union[str, Path], *, fps=50):
    if isinstance(p, Path):
        p = str(p)
    y, sr = load_audio(p)
    lmel1 = get_mel_spectrogram(y, sr, frame_length=1024, fps=fps)
    lmel2 = get_mel_spectrogram(y, sr, frame_length=2048, fps=fps)
    lmel3 = get_mel_spectrogram(y, sr, frame_length=4096, fps=fps)
    max_len = max([lmel1.shape[-1],lmel2.shape[-1],lmel3.shape[-1]])
    if lmel1.shape[-1] < max_len:
        lmel1 = np.pad(lmel1, [[0,0],[0,max_len - lmel1.shape[-1]]])
    if lmel2.shape[-1] < max_len:
        lmel2 = np.pad(lmel2, [[0,0],[0,max_len - lmel2.shape[-1]]])
    if lmel3.shape[-1] < max_len:
        lmel3 = np.pad(lmel3, [[0,0],[0,max_len - lmel3.shape[-1]]])
    return np.stack([lmel1, lmel2, lmel3], axis=0)