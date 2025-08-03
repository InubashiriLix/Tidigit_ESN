import os
import torch
import logging

from train import evaluate
from dataset import TidigitDataset, collate_fn
import torch.utils.data as data
import torch.optim as optim
from config import RAW_TEST_FOLDER_ABS_PATH

from ESN import ESN

LEARNING_RATE = 0.01
MODEL_SAVE_PATH = "best_model.pth"

logging.basicConfig(level=logging.INFO)


test_dataset = TidigitDataset(data_dir=RAW_TEST_FOLDER_ABS_PATH, use_onehot=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESN(
    input_size=13,
    output_size=len(test_dataset.caption_encoder.char_vocab),
    spectral_radius=0.9,
    hidden_size=2000,
).to(DEVICE)
optimizer = optim.Adam(model.W_out.parameters(), lr=LEARNING_RATE)

checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# epoch = checkpoint["epoch"]
# best_loss = checkpoint["best_loss"]

test_dataloader = data.DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn
)
predictions = evaluate(model, test_dataloader)

for i in predictions:
    print(f"instance path: {i[0]}")
    print(f"true caption: {i[1]}")
    print(f"predicted caption: {i[2]}")
