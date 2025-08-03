import numpy as np
import os


def softmax(x: np.ndarray):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


# the input array is desired to be in shape (N, T, C)
# where N stand for batch size, T for sequence length, C for the number of classes
def CTCLoss(
    logits: np.ndarray,
    labels: np.ndarray,
    input_len: int,
    label_len: int,
    blank: int = 0,
):
    # the batch size, we will enumerate all the batchs
    batch_size = logits.shape[0]
    # T will be use in forward and backward computing
    T = logits.shape[1]

    # initialize loss
    loss = np.zeros((batch_size))

    # enumerate the batchs
    for b in range(batch_size):
        # prepare the logits to compute
        logits_seq = logits[b, :, :]
        labels_seq = labels[b, :]

        # log-softmax
        log_probs = softmax(logits_seq)

        # because the CTC can also predict teh blank, we need to add one more class (+1)
        # forward (from start to end) + 1
        alpha = np.zeros((T, len(labels_seq) + 1))
        # 1 + backward (from end to start)
        beta = np.zeros((T, len(labels_seq) + 1))

        for t in range(T):
            # the first second element of the first column should be 1 (start)
            if t == 0:
                alpha[t, 0] = 1 if labels_seq[0] == blank else 0
                alpha[t, 1] = 1 if labels_seq[0] == logits_seq[t] else 0
            else:
                # calculate elements in the columns
                # the first row
                alpha[t, 0] = alpha[t - 1, 0] * (1 if labels_seq[0] == blank else 0)
                # the other rows
                for l in range(1, len(labels_seq) + 1):
                    # transition to the next label
                    alpha[t, l] = (alpha[t - 1, l - 1] + alpha[t - 1, l]) * log_probs[
                        t, labels_seq[l - 1]
                    ]

        for t in reversed(range(T)):
            # backward
            if t == T - 1:
                beta[t, len(labels_seq)] = 1
            else:
                for l in range(len(labels_seq), 0, -1):
                    beta[t, l] = beta[t + 1, l] * log_probs[t, labels_seq[l]]

        loss = np.zeros((T, len(labels_seq)))
        for t in range(T):
            loss += ( alpha[t, :] * beta[t, :] + np.max( alpha ) )

        return loss
    #
