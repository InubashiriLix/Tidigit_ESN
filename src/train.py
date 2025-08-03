import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import os

from ESN import ESN
from dataset import TidigitDataset, collate_fn
from config import RAW_TEST_FOLDER_ABS_PATH

# 训练超参数
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.01
CLIP_GRAD = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型保存路径
MODEL_SAVE_PATH = "best_model.pth"
best_loss = float("inf")

# 构造训练数据集（使用 onehot 标签）
train_dataset = TidigitDataset(use_onehot=True)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# 构造测试数据集，用于评估模型效果
test_dataset = TidigitDataset(data_dir=RAW_TEST_FOLDER_ABS_PATH, use_onehot=True)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn
)

# 构造 ESN 模型（input_size 与 MFCC 的特征维度对应）
model = ESN(
    input_size=13,
    output_size=len(train_dataset.caption_encoder.char_vocab),
    spectral_radius=0.9,
    hidden_size=2000,
).to(DEVICE)

# Xavier 初始化（只初始化 W_out 权重）
nn.init.xavier_uniform_(model.W_out.weight)

# CTC 损失函数
criterion = nn.CTCLoss()
# 只优化 W_out 参数
optimizer = optim.Adam(model.W_out.parameters(), lr=LEARNING_RATE)


def ctc_decode(seq, blank=0):
    """
    对预测的 token 序列进行 CTC 解码：
      1. 折叠连续重复的 token
      2. 移除空白 token（默认 blank=0）
    参数：
      seq: list，预测出的 token 序列（例如 [0, 0, 3, 3, 0, 2, ...]）
      blank: 空白 token 的索引（默认 0）
    返回：
      解码后的 token 序列（例如 [3, 2, ...]）
    """
    new_seq = []
    prev = None
    for token in seq:
        if token == blank:
            prev = token
            continue
        if token != prev:
            new_seq.append(token)
        prev = token
    return new_seq


def evaluate(model: ESN, dataloader):
    """
    对测试数据集进行评估，返回若干条样例的预测结果。
    每个样例包含文件路径、真实字幕和预测字幕。
    """
    # model.eval()
    predictions = []
    with torch.no_grad():
        for mfccs, labels, mfcc_lengths, paths, captions in dataloader:
            mfccs = mfccs.to(DEVICE)
            outputs = model(mfccs)  # (batch_size, T, vocab_size)
            outputs = outputs.permute(1, 0, 2)  # (T, batch_size, vocab_size)
            predicted_ids = torch.argmax(outputs, dim=2)  # (T, batch_size)
            for i in range(predicted_ids.shape[1]):
                raw_seq = predicted_ids[:, i].tolist()
                decoded_seq = ctc_decode(raw_seq, blank=0)
                pred_text = test_dataset.caption_encoder.decode(decoded_seq)
                predictions.append((paths[i], captions[i], pred_text))
            # 这里只取前 5 个样例
            if len(predictions) >= 5:
                break
    return predictions


if __name__ == "__main__":
    log_file_name_index = 0
    while True:
        try:
            os.open("train_log" + str(log_file_name_index) + ".txt", os.O_RDONLY)
            log_file_name_index += 1
        except Exception as e:
            break

    log_file_name = "train_log" + str(log_file_name_index) + ".txt"

    with open(log_file_name, "w") as f:
        print("开始训练...\n")
        f.writelines("start training \n")
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for mfccs, labels, mfcc_lengths, paths, captions in train_dataloader:
                # mfccs: (batch_size, max_T, n_mfcc)
                # labels: 列表，每个元素形状为 (L, vocab_size)（未 pad 的 onehot 标签）
                if mfccs is None:
                    continue

                mfccs = mfccs.to(DEVICE).to(torch.float32)
                # 将 onehot 标签转换为索引序列（1D 张量）
                if (
                    isinstance(labels, list)
                    and len(labels) > 0
                    and labels[0].dim() == 2
                ):
                    new_labels = []
                    for label in labels:
                        indices = []
                        for row in label:
                            if torch.all(row == 0):
                                continue
                            indices.append(int(row.argmax()))
                        new_labels.append(torch.tensor(indices, dtype=torch.long))
                    labels = new_labels

                # 检查 MFCC 是否包含 NaN
                if torch.isnan(mfccs).any():
                    f.writelines(f"MFCC 存在 NaN，跳过 {paths[0]} \n")
                    print(f"MFCC 存在 NaN，跳过 {paths[0]}")
                    continue

                # 前向传播
                outputs = model(mfccs)  # (batch_size, seq_len, vocab_size)
                outputs = outputs.permute(1, 0, 2)  # (T, batch_size, vocab_size)

                if torch.isnan(outputs).any():
                    f.writelines("模型输出 NaN，跳过该 batch \n")
                    print("模型输出 NaN，跳过该 batch")
                    continue

                # 对输出做 log_softmax（CTC 损失要求对数概率）
                outputs = F.log_softmax(outputs, dim=2)

                # 使用 collate_fn 返回的真实时间步数（input_lengths）
                input_lengths = mfcc_lengths.to(DEVICE)  # (batch_size,)

                # 计算每个样本目标标签的真实长度，并将所有标签拼接成 1D 张量
                target_lengths = torch.tensor(
                    [len(lbl) for lbl in labels], dtype=torch.long, device=DEVICE
                )
                targets = torch.cat(labels).to(DEVICE)

                # 检查目标长度总和是否合理
                if target_lengths.sum() > input_lengths.sum():
                    f.writelines(
                        "警告: target_lengths 超过 input_lengths，跳过该 batch \n"
                    )
                    print("警告: target_lengths 超过 input_lengths，跳过该 batch")
                    continue

                loss = criterion(outputs, targets, input_lengths, target_lengths)

                if torch.isnan(loss):
                    f.writelines("Loss 变成 NaN，跳过该 batch \n")
                    print("Loss 变成 NaN，跳过该 batch")
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.W_out.parameters(), max_norm=CLIP_GRAD
                )
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")
            f.writelines(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f} \n")

            # 保存最优模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                print("asdhfilashdl")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    },
                    MODEL_SAVE_PATH,
                )
                print(
                    f"最优模型已保存！Loss: {best_loss:.4f} | 保存至: {MODEL_SAVE_PATH}"
                )
                f.writelines(
                    f"最优模型已保存！Loss: {best_loss:.4f} | 保存至: {MODEL_SAVE_PATH} \n"
                )

            # 每个 epoch 后对测试集进行简单评估，展示部分预测结果

            # counting correct rate
            correct_count = 0
            total_count = 0
            preds = evaluate(model, test_dataloader)
            # print("【测试集预测样例】")
            for path, true_caption, pred_caption in preds:
                # print(f"文件路径: {path}")
                # print(f"真实字幕: {true_caption}")
                # print(f"预测字幕: {pred_caption}")
                # print("-" * 40)

                correct_count = (
                    correct_count + 1 if true_caption == pred_caption else correct_count
                )
                for index, char in enumerate(true_caption):
                    if index < len(pred_caption):
                        correct_count = (
                            correct_count + 1
                            if char == pred_caption[index]
                            else correct_count
                        )
                total_count += len(true_caption)

            print(f"正确率: {correct_count}/{total_count}")
            f.writelines(f"正确率: {correct_count}/{total_count} \n")

            preds = evaluate(model, train_dataloader)
            print("【训练集预测样例】")
            for path, true_caption, pred_caption in preds:
                print(f"文件路径: {path}")
                print(f"真实字幕: {true_caption}")
                print(f"预测字幕: {pred_caption}")
                print("-" * 40)

        print("训练完成！")
        f.writelines("训练完成！\n")
