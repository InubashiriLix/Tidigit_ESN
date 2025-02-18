import numpy as np
import scipy
import librosa

import scipy.fftpack

# NOTE: well, pyright LSP force me to keep an eye on the type of each thing (variable, function ...)
# so, if you notice something like checking None value, please ignore my lost of elegent.


def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    pre emphasis to maintain the high frequency part of the signal
    signal: 1D numpy array, the data of the audio signal
    coeff: emphasis coefficiency，usually between 0.95~0.98.
    """
    emphasized_signal: np.ndarray = np.append(
        signal[0], signal[1:] - coeff * signal[:-1]
    )
    return emphasized_signal


def framing(
    signal: np.ndarray, frame_size: float, hop_size: float, fs: int
) -> np.ndarray:
    """
    framing the signal
    - signal: signal after pre emephasis
    - frame_size: length of a frame (second)
    - hop_size: hop size: the distance between two frames (second)
    - fs: sampling rate

    @ return
    - frames: shape (num_frames, frame_len) 2D array，each row is a frame
    """
    frame_len = int(round(frame_size * fs))  # number of sampling points per frame
    hop_len = int(round(hop_size * fs))  # numbers of sampling points per hop
    num_frames = 1 + (len(signal) - frame_len) // hop_len

    frames = np.zeros((num_frames, frame_len))  # create a 2D array to store the frames
    # store the frames
    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frames[i, :] = signal[start:end]
    return frames


def apply_hamming(frames: np.ndarray) -> np.ndarray:
    """
    apply hamming window to each frame
    frames: (num_frames, frame_len)
    return
    - windowed_frames: the frames after applying hamming window
    """
    num_frames, frame_len = frames.shape
    window = np.hamming(frame_len)
    return frames * window


def compute_magnitude_spectrum(frames, NFFT=512):
    """
    apply FFT to each frame and get the magnitude spectrum
    - frames: shape (num_frames, frame_len)
    - NFFT: FFT points

    return：
    - mag_spectra: shape (num_frames, NFFT//2 + 1)
    """
    num_frames, frame_len = frames.shape
    mag_spectra = np.zeros((num_frames, NFFT // 2 + 1))

    for i in range(num_frames):
        spectrum = np.fft.rfft(frames[i], n=NFFT)
        mag_spectra[i, :] = np.abs(spectrum)
    return mag_spectra


def hz_to_mel(hz):
    """
    using formula 2595 * log10(1 + hz/700)
    """
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filter_bank(num_filters: int, NFFT: int, fs: int):
    """
    construct Mel filter bank
    - num_filters: filter num
    - NFFT: FFT points
    - fs: sampling rate
    return
    - fbank: filter bank, shape: (num_filters, NFFT//2 + 1)
    """
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(fs / 2)

    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)

    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((NFFT + 1) * hz_points / fs).astype(int)

    fbank = np.zeros((num_filters, NFFT // 2 + 1))
    for i in range(1, num_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for j in range(left, center):
            fbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbank[i - 1, j] = (right - j) / (right - center)

    return fbank


def log_mel_spectrum(mag_spectra, fbank):
    """
    multipy the magnitude spectrum with the mel filter bank, then return the log value
    - mag_spectra: (num_frames, NFFT//2+1)
    - fbank: (num_filters, NFFT//2+1)
    return：
    - log_mel: (num_frames, num_filters)
    """
    mel_spectra = np.dot(mag_spectra, fbank.T)
    # well, in case of log 0
    mel_spectra = np.where(mel_spectra == 0, np.finfo(float).eps, mel_spectra)
    return np.log(mel_spectra)


def dct_features(log_mel_spectra, num_ceps=13):
    """
    apply DCT to log mel spectrum to get MFCC
    - log_mel_spectra: (num_frames, num_filters)
    - num_ceps: number of cepstrum coefficients
      return
    - mfcc: (num_frames, num_ceps)
    """
    # use type=2, norm='ortho' to fit the common config for DCT
    return scipy.fftpack.dct(log_mel_spectra, type=2, axis=1, norm="ortho")[
        :, :num_ceps
    ]


def compute_mfcc(
    signal,
    fs,
    frame_size=0.025,
    hop_size=0.01,
    pre_emph=0.97,
    NFFT=512,
    num_filters=40,
    num_ceps=13,
):
    # connect all the steps above to get the MFCC
    emphasized_signal = pre_emphasis(signal, coeff=pre_emph)
    frames = framing(emphasized_signal, frame_size, hop_size, fs)
    windowed_frames = apply_hamming(frames)
    mag_spectra = compute_magnitude_spectrum(windowed_frames, NFFT=NFFT)
    fbank = mel_filter_bank(num_filters, NFFT, fs)
    log_mel = log_mel_spectrum(mag_spectra, fbank)
    mfcc = dct_features(log_mel, num_ceps=num_ceps)
    return mfcc


if __name__ == "__main__":
    # load the data
    import os
    from config import RAW_TRAIN_FOLDER_ABS_PATH, REGEX_PATTERN
    import logging
    import re  # I hate regex
    import librosa

    logging.basicConfig(level=logging.INFO)

    audio_mfcc_list: list[dict] = []

    for audio in os.listdir(RAW_TRAIN_FOLDER_ABS_PATH):
        # check format
        if not audio.endswith(".wav"):
            logging.warning(f"Skip {audio}, cause it's not a wav file")

        # check the audio file
        try:
            # get reletive path
            audio_abs_path = os.path.join(RAW_TRAIN_FOLDER_ABS_PATH, audio)

            # get the caption of the audio
            # a file should like "jr_2a.wav"
            # and "2" is the true caption we need
            # and filw with name "st_oa.wav"
            # is "o"
            match_result = re.match(REGEX_PATTERN, audio)
            audio_caption = match_result.group(1) if match_result else "Error"

            # get the mfcc value for single audio
            signal, fs = librosa.load(audio_abs_path, sr=None)
            mfcc = compute_mfcc(signal, fs)

            # append the caption and mfcc to the dictionary
            temp_audio_mfcc_dict = {audio_caption: mfcc}
            audio_mfcc_list.append(temp_audio_mfcc_dict)

            logging.info(f"Loading {audio} complete")
        except Exception as e:
            logging.warning(f"Error loading {audio}: {e}")

    for audio, _ in zip(audio_mfcc_list, range(5)):
        for key, value in audio.items():
            # logging.info(f"Caption: {key}, MFCC: {value} ")
            print(value.shape)
