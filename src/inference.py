import os
import torch
import logging

from train import evaluate
from dataset import TidigitDataset, collate_fn
from config import RAW_TEST_FOLDER_ABS_PATH

logging.basicConfig(level=logging.INFO)
logging.error("the inference has bug now")

model = torch.load("best_model.pth")
predictions = evaluate(model, TidigitDataset(RAW_TEST_FOLDER_ABS_PATH))

for i in predictions:
    print(f"instance path: {i[0]}")
    print(f"true caption: {i[1]}")
    print(f"predicted caption: {i[2]}")
