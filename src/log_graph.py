import re
import matplotlib.pyplot as plt


def parse_log(log_text):
    epochs = []
    losses = []
    accuracies = []

    # 正则表达式匹配 Epoch、Loss 和 正确率
    epoch_pattern = re.compile(r"Epoch \[(\d+)/\d+\], Loss: ([\d\.]+)")
    accuracy_pattern = re.compile(r"正确率: (\d+)/(\d+)")

    for line in log_text.split("\n"):
        epoch_match = epoch_pattern.search(line)
        accuracy_match = accuracy_pattern.search(line)

        if epoch_match:
            epoch = int(epoch_match.group(1))
            loss = float(epoch_match.group(2))
            epochs.append(epoch)
            losses.append(loss)

        if accuracy_match:
            correct = int(accuracy_match.group(1))
            total = int(accuracy_match.group(2))
            accuracy = correct / total * 100  # 转换为百分比
            accuracies.append(accuracy)

    return epochs, losses, accuracies


def plot_metrics(epochs, losses, accuracies):
    plt.figure(figsize=(10, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker="o", linestyle="-", label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs. Loss")
    plt.legend()

    # 绘制 正确率 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker="s", linestyle="-", color="g", label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Epoch vs. Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 你的训练日志
log_file_name = "train_log4.txt"
with open(log_file_name, "r") as f:
    log_text = f.read()

# 解析日志并绘制曲线
epochs, losses, accuracies = parse_log(log_text)
plot_metrics(epochs, losses, accuracies)
