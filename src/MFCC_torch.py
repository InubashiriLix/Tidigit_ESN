import torchaudio
import torchaudio.transforms as transforms
import logging

from config import (
    RAW_TRAIN_FOLDER_ABS_PATH,
    RAW_TEST_FOLDER_ABS_PATH,
    target_sample_rate,
    DEBUG,
)
from utils import get_audio_abs_path_list, get_caption

from config import torchaudio_mfcc_settings

# setup the transforms
transform_mfcc = transforms.MFCC(**torchaudio_mfcc_settings)

# set up the logging
logging.basicConfig(level=logging.INFO)

# WARNING: please check and update the set setting in case of saving the data into the wrong dir
is_train_set: bool = True
raw_data_folder_abs_path = (
    RAW_TRAIN_FOLDER_ABS_PATH if is_train_set else RAW_TEST_FOLDER_ABS_PATH
)


if __name__ == "__main__":
    dataset_path_list = get_audio_abs_path_list(raw_data_folder_abs_path, ".wav")
    if not dataset_path_list:
        logging.error("No data found in the dataset")
        exit(1)

    # process with data and put them into the files
    mfccs = []
    for audio_abs_path in dataset_path_list:
        try:
            # get the caption
            caption = get_caption(audio_abs_path)
            if caption is None:
                continue
            # load the audio
            print(audio_abs_path)
            waveform, sample_rate = torchaudio.load(audio_abs_path, normalize=True)
            # if the waveform is dual channel, we just need one channel
            if waveform.shape[0] >= 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            # if the sample rate is not 16k, we need to resample it
            if sample_rate != target_sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=target_sample_rate
                )(waveform)
            # apply MFCC
            mfcc = transform_mfcc(waveform)
            mfccs.append(mfcc[0])
            if DEBUG:
                logging.info(f"shape: {audio_abs_path} shape: {mfcc.shape}")
        except Exception as e:
            logging.error(f"Error in processing {audio_abs_path}")
            logging.info(e)
            continue
