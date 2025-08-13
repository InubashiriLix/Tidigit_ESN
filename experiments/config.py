# this a configuration file for the Tidigit_Train.py script
import os

DEBUG: bool = True

torch_resource_format = "pt"
# avaliable: pt, pth, npy

# the training set path
RAW_RESOURCES_FOLDER = "data/raw"

RAW_TRAIN_FOLDER_NAME = "tidigits_train"
RAW_TEST_FOLDER_NAME = "tidigits_test"


# TRAIN_ABS_PATH =
def get_path_abs_dataset_folder(dataset_folder_name: str) -> str:
    root_abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_abs_path, RAW_RESOURCES_FOLDER, dataset_folder_name)


RAW_TRAIN_FOLDER_ABS_PATH = (
    "/home/inubashiri/01_ESN/TidigitClassification/data/raw/tidigits_train"
)
RAW_TEST_FOLDER_ABS_PATH = (
    "/home/inubashiri/01_ESN/TidigitClassification/data/raw/tidigits_test/"
)


# regex part
REGEX_PATTERN = r"^.*?_(.*?)[a-z].wav"

# For torchaudio MFCC
# Sample rate of audio signal
target_sample_rate: int = 16_000
sample_rate: int = 16000

# Number of mfc coefficients to retain.
# NOTE: this value should less than n_mels in melkwargs
# (feature of DCT?) (you can config it more than 13 but the part more than 13 contains that noise)
n_mfcc: int = 13

# type of DCT (discrete cosine transform) to use.
dct_type: int = 2

# norm: norm to use
norm: str = "ortho"

# log_mels: whether to use log-mel spectrograms instead of db-scaled
log_mels: bool = True

# melkwargs: arguments for MelSpectrogram
melkwargs = {"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}

torchaudio_mfcc_settings = {
    "sample_rate": sample_rate,
    "n_mfcc": n_mfcc,
    "dct_type": dct_type,
    "norm": norm,
    "log_mels": log_mels,
    "melkwargs": melkwargs,
}


# for librosa MFCC
fs = 16000
frame_size = 0.025
hop_size = 0.01
pre_emph = 0.97
NFFT = 512
num_filter = 40
num_ceps = 13
