import os
import torch
import torch.nn.functional as F
import torchaudio

from ESN import ESN  # 请确保 ESN 类实现正确且可导入
from dataset import TidigitDataset, collate_fn  # 请确保数据集类和 collate_fn 可导入
from config import (
    target_sample_rate,
    RAW_TEST_FOLDER_ABS_PATH,
)  # 采样率及测试数据集路径
from utils import get_caption  # 获取真实字幕的辅助函数

# ---------------------------
# 设备与模型相关设置
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 模型文件 {MODEL_PATH} 不存在，请先训练模型。")

# 加载模型（你也可以设置 weights_only=True 来提高安全性）
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = ESN(input_size=13, output_size=36, spectral_radius=0.6).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ---------------------------
# 载入测试数据集
# ---------------------------
test_dataset = TidigitDataset(RAW_TEST_FOLDER_ABS_PATH)
# 注意：collate_fn 返回的格式为 (mfccs, labels, mfcc_lengths, paths, captions)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)


# ---------------------------
# CTC 解码函数
# ---------------------------
def ctc_decode(seq, blank=0):
    """
    对预测的 token 序列进行 CTC 解码：
      - 折叠连续重复的 token
      - 移除空白 token（即 blank）
    参数：
      seq: list，预测出的 token 序列（例如 [0, 0, 3, 3, 0, 2, ...]）
      blank: 空白 token 的索引（默认 0）
    返回：
      解码后的 token 列表（例如 [3, 2, ...]）
    """
    new_seq = []
    prev = None
    for token in seq:
        # 若当前 token 为空白，直接更新 prev 并跳过
        if token == blank:
            prev = token
            continue
        if token != prev:
            new_seq.append(token)
        prev = token
    return new_seq


# ---------------------------
# 批量预测（测试数据集）
# ---------------------------
def predict():
    print("\n🚀 正在对测试数据集进行预测...\n")
    with torch.no_grad():
        for mfccs, labels, _, paths, captions in test_dataloader:
            mfccs = mfccs.to(DEVICE)
            # 前向传播得到输出 shape: (batch_size, T, vocab_size)
            outputs = model(mfccs)
            # 转置为 (T, batch_size, vocab_size)
            outputs = outputs.permute(1, 0, 2)
            # 如果需要，可以先做 log_softmax，但对 argmax 并不影响
            # outputs = F.log_softmax(outputs, dim=2)
            predicted_ids = torch.argmax(outputs, dim=2)  # shape: (T, batch_size)

            predicted_texts = []
            for i in range(predicted_ids.shape[1]):  # 遍历 batch 内每个样本
                raw_seq = predicted_ids[:, i].tolist()
                # 使用 CTC 解码：折叠重复并移除空白 token（默认 blank=0）
                decoded_seq = ctc_decode(raw_seq, blank=0)
                # 调用字符解码器将 token 序列转换为字符串
                predicted_text = test_dataset.caption_encoder.decode(decoded_seq)
                predicted_texts.append(predicted_text)

            # 输出该 batch 的结果（这里只展示一个 batch）
            for i in range(len(predicted_texts)):
                print(f"🎯 真实字幕: {captions[i]}")
                print(f"🤖 预测字幕: {predicted_texts[i]}")
                print(f"📂 文件路径: {paths[i]}")
                print("-" * 40)
            break  # 只处理一个 batch，防止输出过多


# ---------------------------
# 单个文件预测
# ---------------------------
def predict_single(file_path):
    print(f"\n🚀 正在对单个文件进行预测: {file_path}\n")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")
    with torch.no_grad():
        # 读取音频
        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        # 若为多通道，取平均转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # 若采样率不符，则进行重采样
        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            )(waveform)
        # 计算 MFCC 特征，输出形状 (T, n_mfcc)
        transform_mfcc = torchaudio.transforms.MFCC(
            sample_rate=target_sample_rate, n_mfcc=13
        )
        mfcc = transform_mfcc(waveform).squeeze(0).transpose(0, 1)
        mfcc = mfcc.unsqueeze(0).to(DEVICE)  # shape: (1, T, n_mfcc)

        outputs = model(mfcc)  # shape: (1, T, vocab_size)
        outputs = outputs.permute(1, 0, 2)  # shape: (T, 1, vocab_size)
        predicted_ids = torch.argmax(outputs, dim=2)  # shape: (T, 1)
        raw_seq = predicted_ids[:, 0].tolist()
        decoded_seq = ctc_decode(raw_seq, blank=0)
        predicted_text = test_dataset.caption_encoder.decode(decoded_seq)

        real_caption = get_caption(file_path)
        print(f"🎯 真实字幕: {real_caption if real_caption else '未知'}")
        print(f"🤖 预测字幕: {predicted_text}")
        print(f"📂 文件路径: {file_path}")
        print("-" * 40)


# ---------------------------
# 主函数：运行预测
# ---------------------------
if __name__ == "__main__":
    # 先对整个测试数据集进行预测
    predict()

    # 再对单个文件进行预测（请替换为实际文件路径）
    # test_file = "测试音频文件路径.wav"  # 替换为你的测试文件路径
    # predict_single(test_file)
