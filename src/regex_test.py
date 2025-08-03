import re
import os
from config import RAW_TRAIN_FOLDER_ABS_PATH as TRAIN_PATH

file_name_list = os.listdir(TRAIN_PATH)

count = 0
for file_name in file_name_list:
    if file_name.endswith(".wav"):
        count += 1
    else:
        print(f"{file_name} is not a .wav file.")

if count == len(file_name_list):
    print("All files are .wav files.")

pattern = r"^.*?_(.*?)[a-z].wav"

for file_name in file_name_list:
    match = re.match(pattern, file_name)
    if match:
        audio_caption = match.group(1)  # 获取匹配的数字部分
        print(f"origin: {file_name}, caption: {audio_caption}")
    else:
        print(f"not founded {file_name}")
