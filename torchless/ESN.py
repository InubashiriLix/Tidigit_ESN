import numpy as np


class ESN:
    def __init__(
        self, input_size, output_size, hidden_size=100, spectral_radius=0.9, seed=None
    ):
        if seed is None:
            np.random.seed(seed)

        self.hidden_size = hidden_size

        # initialize weights
        w = np.random.rand(hidden_size, hidden_size) - 0.5

        # get the feature values
        eigenvalues, _ = np.linalg.eig(w)
        max_eig = np.max(np.abs(eigenvalues))

        # adjust the spectral_radius
        self.W_res = w * (spectral_radius / max_eig)

        # initialize input weights
        self.W_in = np.random.rand(hidden_size, input_size) - 0.5

        # initialize the output weights
        self.W_out = np.random.rand(output_size, hidden_size)
        self.b_out = np.zeros((output_size,))

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        return shape (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, input_size = x.shape

        h = np.zeros((batch_size, self.hidden_size))
        outputs = []
        for t in range(seq_len):
            u = x[:, t, :]
            h = np.tanh(np.matmul(h, self.W_res.T) + np.matmul(u, self.W_in.T))
            out = np.matmul(h, self.W_out.T) + self.W_out
            outputs.append(np.expand_dims(out, axis=1))

        return np.concatenate(outputs, axis=1)


if __name__ == "__main__":
    input_size = 10
    outpus_size = 2
    hidden_size = 100
    seq_len = 10
    batch_size = 3

    # random inputs
    x = np.random.rand(batch_size, seq_len, input_size)

    # initialize the ESN
    esn = ESN(input_size, outpus_size, hidden_size)

    outputs = esn.forward(x)
    print("outputs shape:", outputs.shape)
