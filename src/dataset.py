import os
from typing import Any
import torch
import torch.utils.data as data
import torchaudio.transforms as transforms
import torchaudio

import logging


from utils import get_audio_abs_path_list, get_caption

from config import (
    RAW_TRAIN_FOLDER_ABS_PATH,
    RAW_TEST_FOLDER_ABS_PATH,
    target_sample_rate,
    torchaudio_mfcc_settings,
)

logging.basicConfig(level=logging.INFO)

use_train_set = True

transform_mfcc = transforms.MFCC(**torchaudio_mfcc_settings)


class CaptionEncoder:
    def __init__(self):
        # NOTE: 记得在编码之前将 caption 转为小写
        self.char_vocab = {
            ch: idx for idx, ch in enumerate("0123456789abcdefghijklmnopqrstuvwxyz")
        }
        self.idx_to_char = {idx: ch for ch, idx in self.char_vocab.items()}
        self.char_vocal_size = len(self.char_vocab)  # 字符表的大小

    def encode(self, caption: str):
        """将 caption 转换为索引序列"""
        return torch.tensor(
            [self.char_vocab[ch] for ch in caption if ch in self.char_vocab],
            dtype=torch.long,
        )

    def encode_onehot(self, text: str, vocab_size: int = 36):
        """
        将 caption 转换为 onehot 矩阵，形状为 (L, vocab_size)。
        如果未指定 vocab_size，则使用实际的字符表大小。
        """
        if vocab_size is None:
            vocab_size = self.char_vocal_size
        label_tensor = torch.zeros(len(text), vocab_size)
        for i, ch in enumerate(text):
            idx = self.char_vocab.get(ch, None)
            if idx is not None:
                label_tensor[i, idx] = 1
        return label_tensor

    def decode_onehot(self, onehot_matrix: torch.Tensor) -> str:
        """
        将 onehot 矩阵解码为字符串。
        对于每一行，如果该行全为 0（例如 padding 部分），则跳过；
        否则取该行最大值所在的索引转换为字符。
        """
        decoded_chars = []
        for row in onehot_matrix:
            # 若该行全为 0，则认为是 padding，跳过该行
            if torch.all(row == 0):
                continue
            idx = int(row.argmax())
            if idx in self.idx_to_char:
                decoded_chars.append(self.idx_to_char[idx])
        return "".join(decoded_chars)

    def decode(self, indices):
        """将索引序列转换回字符串（跳过填充的 `-1`）"""
        return "".join(
            [self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char]
        )

    def __call__(self, caption: str, onehot=False):
        if onehot:
            return self.encode_onehot(caption)
        return self.encode(caption)


class TidigitDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str | list[str] = RAW_TRAIN_FOLDER_ABS_PATH
        if use_train_set
        else RAW_TEST_FOLDER_ABS_PATH,
        extension: str = ".wav",
        use_onehot: bool = False,
        transformer: transforms.MFCC | Any = transform_mfcc,
        caption_encoder=CaptionEncoder(),
    ):
        # basic config
        # NOTE: the data_dir is the ABS root dir of the dataset
        self.data_dir: str | list[str] = data_dir
        # NOTE: the transformer should be a function / class that can be called with a waveform tensor and return a mfcc tensor
        self.transformer = transformer
        # NOTE: the caption_encoder should be a class or function that can be called with a caption string and return a tensor
        self.caption_encoder = caption_encoder
        # NOTE: use_onehot is a flag to determine whether to use onehot encoding for the caption
        self.use_onehot: bool = use_onehot
        self.extension = extension

        self.data_cap_mfcc_dict = {}
        # NOTE: format of the data mfcc_dict
        # self.data_cap_mfcc_dict[audio_abs_path] = {
        #     "mfcc": mfcc,  # (T, n_mfcc)
        #     "label": label,  # (L,) or (L, vocab_size)
        #     "caption": caption,
        # }

        # load the data when the class is initialized
        self._load_data(self.extension)
        if not self.data_cap_mfcc_dict:
            logging.error("No data loaded")
            raise Exception("No data loaded")

        self.dataset_path_list: list[str]

    def __getitem__(self, index: int):
        # 直接从预加载 dict 里取
        audio_path = self.dataset_path_list[index]
        info = self.data_cap_mfcc_dict[audio_path]

        return (info["label"], info["mfcc"], audio_path, info["caption"])

    def _load_data(self, extension: str):
        """
        一次性加载所有音频并处理，存入 self.data_cap_mfcc_dict
        包括：转单声道、采样率统一、MFCC 计算、归一化以及 caption 编码
        """
        self.dataset_path_list = get_audio_abs_path_list(self.data_dir, extension)

        for audio_abs_path in self.dataset_path_list:
            try:
                waveform, sample_rate = torchaudio.load(audio_abs_path, normalize=True)
                # 若为多通道，取平均转为单声道
                if waveform.shape[0] >= 2:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # 采样率转换
                if sample_rate != target_sample_rate:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=target_sample_rate
                    )(waveform)

                # 计算 MFCC，原始形状为 (1, n_mfcc, T)
                mfcc = self.transformer(waveform).squeeze(0)  # -> (n_mfcc, T)
                # 转置为 (T, n_mfcc) 方便后续处理
                mfcc = mfcc.transpose(0, 1)

                # 对 MFCC 进行零均值、单位方差归一化（按每个特征维度计算均值和标准差）
                mean = mfcc.mean(dim=0, keepdim=True)
                std = mfcc.std(dim=0, keepdim=True) + 1e-6  # 防止除 0
                mfcc = (mfcc - mean) / std

                file_name = os.path.basename(audio_abs_path)
                caption: str = get_caption(file_name).lower()
                label = (
                    self.caption_encoder.encode_onehot(caption)
                    if self.use_onehot
                    else self.caption_encoder.encode(caption)
                )

                # 存储结果
                self.data_cap_mfcc_dict[audio_abs_path] = {
                    "mfcc": mfcc,  # (T, n_mfcc)
                    "label": label,  # (L,) 或 (L, vocab_size)
                    "caption": caption,
                }

            except Exception as e:
                logging.error(f"Error in processing {audio_abs_path}: {e}")
                raise e

    def __len__(self):
        return len(self.data_cap_mfcc_dict)

    def __str__(self):
        return f"Dataset with {len(self.data_cap_mfcc_dict)} samples"

    def __sizeof__(self) -> int:
        return len(self.data_cap_mfcc_dict)


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    labels, mfccs, paths, captions = zip(*batch)
    # 计算每个样本 MFCC 的真实长度（时间步数）
    mfcc_lengths = torch.tensor([m.shape[0] for m in mfccs], dtype=torch.long)
    # 对 MFCCs 进行 pad，使得形状一致
    mfccs_padded = pad_sequence(mfccs, batch_first=True, padding_value=0)

    # 对于标签：
    # 如果标签为一维（索引序列），则 pad 后返回；
    # 如果为二维（onehot 矩阵），直接以列表形式返回，保持各自的原始长度
    if labels[0].dim() == 1:
        labels_processed = pad_sequence(labels, batch_first=True, padding_value=-1)
    else:
        labels_processed = list(labels)

    return mfccs_padded, labels_processed, mfcc_lengths, paths, captions


if __name__ == "__main__":
    # 若希望使用 onehot 编码，可以将 use_onehot 参数设为 True
    dataset = TidigitDataset(use_onehot=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    for mfccs, labels, paths, captions in dataloader:
        print(f"MFCC 形状: {mfccs.shape}")  # (batch_size, max_T, n_mfcc)
        print(
            f"标签 shape: {labels.shape}"
        )  # 若 onehot 则为 (batch_size, max_label_length, vocab_size)
        print(f"示例标签: {labels[0]}")
        print(f"文件路径: {paths[0]}")
        print(f"caption: {captions[0]}")

        # 示例：使用 CaptionEncoder 的 decode_onehot 解码 onehot 标签
        decoded_caption = dataset.caption_encoder.decode_onehot(labels[0])
        print(f"解码后的 caption: {decoded_caption}")
        break
