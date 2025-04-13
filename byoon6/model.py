import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder Network: Maps raw features to a normalized embedding space.
class TsclEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.5):
        """
        Args:
            input_dim (int): Number of input features at each time step.
            hidden_dim (int): Size of the hidden layer.
            embedding_dim (int): Dimension of the output embedding.
            dropout (float): Dropout rate.
        """
        super(TsclEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.do = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.do(x)
        x = self.fc2(x)
        # Normalize to lie on the hypersphere (L2 normalization)
        x = F.normalize(x, p=2, dim=1)
        return x

# Predictor Network: Maps embeddings to output logits (class probabilities).
class TsclPredictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int):
        """
        Args:
            embedding_dim (int): Dimension of the input embedding.
            hidden_dim (int): Hidden layer size.
            num_classes (int): Number of classes for classification.
        """
        super(TsclPredictor, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)
        return x

# Temporal Network: Predicts the next time step embedding from the current embedding.
class TsclTemporalNet(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Args:
            embedding_dim (int): Dimension of the input embeddings.
            hidden_dim (int): Hidden layer size.
        """
        super(TsclTemporalNet, self).__init__()
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x
