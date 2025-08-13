import os
from config import REGEX_PATTERN, RAW_TRAIN_FOLDER_ABS_PATH, RAW_TEST_FOLDER_ABS_PATH
import logging
import re

logging.basicConfig(level=logging.INFO)


def get_audio_abs_path_list(
    dataset_path: str | list[str], format_list: tuple | str
) -> list[str]:
    """
    return the list of audio abs path
    @param dataset_path: str the path to the dataset folder, abs path better
    @param format_list: tuple | str the format of the audio file, like ".wav", ".mp3", ".flac", the tidigit dataset is ".wav"
    @return the list of audio abs path, if None, then return None
    """
    audio_abs_pth_list: list[str] = []

    if isinstance(dataset_path, list):
        for pth in dataset_path:
            for audio in os.listdir(pth):
                # check the format
                if audio.endswith(format_list):
                    audio_abs_pth_list.append(os.path.join(pth, audio))
    elif isinstance(dataset_path, str):
        for audio in os.listdir(dataset_path):
            # check the format
            if audio.endswith(format_list):
                audio_abs_pth_list.append(os.path.join(dataset_path, audio))

    if audio_abs_pth_list.__len__() == 0:
        raise ValueError(f"no audio file found in {dataset_path}")

    return audio_abs_pth_list


raw_train_audio_abs_path_list = get_audio_abs_path_list(
    RAW_TRAIN_FOLDER_ABS_PATH, ".wav"
)
raw_test_audio_abs_path_list = get_audio_abs_path_list(RAW_TEST_FOLDER_ABS_PATH, ".wav")


def get_caption(audio_file_name: str) -> str:
    """
    return the caption of the audio file
    @param audio_file_name: str filename only
    @return the audio caption or the None when no content can match
    """
    regex_result: re.Match | None = re.match(REGEX_PATTERN, audio_file_name)
    if regex_result:
        return regex_result.group(1)
    else:
        logging.error(f"error in regex matching for {audio_file_name}")
        raise (ValueError(f"error in regex matching for {audio_file_name}"))


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    labels, mfcc, paths = zip(*batch)
    mfccs = pad_sequence(mfcc, batch_first=True, padding_value=-1)

    return mfccs, labels, paths


#
# if __name__ == "__main__":
#     from config import RAW_TRAIN_FOLDER_ABS_PATH
#
#     ls = get_audio_abs_path_list(RAW_TRAIN_FOLDER_ABS_PATH, ".wav")
#
#     print(ls)
