import math
import os
import numpy as np
from torch.utils.data.datapipes import datapipe

from ESN import ESN
from dataset import CaptionEncoder, TidigitDataset, collate_fn
from dataloader import DataLoader

from config import (
    RAW_TEST_FOLDER_ABS_PATH,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    SR,
)

best_loss = float("inf")

# 构造训练数据集（使用 onehot 标签）
train_dataset = TidigitDataset(use_onehot=True)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# 构造测试数据集，用于评估模型效果
test_dataset = TidigitDataset(data_dir=RAW_TEST_FOLDER_ABS_PATH, use_onehot=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn
)


model = ESN(
    input_size=13,
    output_size=len(train_dataset.caption_encoder.char_vocab),
    spectral_radius=SR,
    hidden_size=2000,
)


def xavier(input_size, output_size):
    scale = 1 / max(1.0, (input_size + output_size) / 2.0)
    limit = math.sqrt(3.0 * scale)


xavier(*model.W_out.shape)

for x in train_dataloader:
    model.forward(x["input"])
