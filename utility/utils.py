# Third Party
import librosa
import numpy as np
import torch

# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    len_gt = 128000
    true_len = len(wav)
    if true_len <= len_gt:
        diff = len_gt - true_len
        dum = np.zeros((1,diff))
        new_wav = np.concatenate((wav,dum[0]))
        wav = new_wav
    else:
        new_wav = wav[0:128000]
        wav = new_wav
    return wav

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    #linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    melbank = librosa.feature.melspectrogram(wav,hop_length=hop_length,win_length=win_length,n_mels=40)
    return melbank


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    spec_delta = librosa.feature.delta(linear_spect)
    melbank_delta = np.concatenate((linear_spect,spec_delta))
    mag, _ = librosa.magphase(melbank_delta)  # magnitude
    mag_T = mag
    freq, time = mag_T.shape
    #if mode == 'train':
    #    randtime = np.random.randint(0, time-spec_len)
    #    spec_mag = mag_T[:, randtime:randtime+spec_len]
    #else:
    #    spec_mag = mag_T
    spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    ret_spec = (spec_mag - mu) / (std + 1e-5)
    #ret_spec = ret_spec.reshape(1,ret_spec.shape[0],ret_spec.shape[1])
    ret_spec = ret_spec.T
    #sub_sampled = ret_spec[np.asarray([i for i in range(0,3126,3)])]
    return ret_spec


