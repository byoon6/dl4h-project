import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder Network: Maps raw features to a normalized embedding space.
class TemporalSCLEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.5):
        """
        Args:
            input_dim (int): Number of input features at each time step.
            hidden_dim (int): Size of the hidden layer.
            embedding_dim (int): Dimension of the output embedding.
            dropout (float): Dropout rate.
        """
        super(TemporalSCLEncoder, self).__init__()
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
class TemporalSCLPredictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        """
        Args:
            embedding_dim (int): Dimension of the input embedding.
            hidden_dim (int): Hidden layer size.
            num_classes (int): Number of classes for classification.
            dropout (float): Dropout rate.
        """
        super(TemporalSCLPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        return self.predictor(x)

# Temporal Network: Predicts the next time step embedding from the current embedding.
class TemporalSCLTemporalNet(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Args:
            embedding_dim (int): Dimension of the input embeddings.
            hidden_dim (int): Hidden layer size.
            dropout (float): Dropout rate.
        """
        super(TemporalSCLTemporalNet, self).__init__()
        self.temporal_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, embedding):
        # embedding shape: (batch_size, embedding_dim)
        return self.temporal_net(embedding)

# Unified Model: Combines encoder, temporal network, and predictor.
class TemporalSCLModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_hidden_dim: int,
                 embedding_dim: int,
                 temporal_hidden_dim: int,
                 predictor_hidden_dim: int,
                 num_classes: int,
                 dropout: float = 0.1):
        """
        Args:
            input_dim (int): Number of input features per time step.
            encoder_hidden_dim (int): Hidden layer size for the encoder.
            embedding_dim (int): Output embedding dimension.
            temporal_hidden_dim (int): Hidden layer size for the temporal network.
            predictor_hidden_dim (int): Hidden layer size for the predictor network.
            num_classes (int): Number of classes for the final prediction.
            dropout (float): Dropout rate for all networks.
        """
        super(TemporalSCLModel, self).__init__()
        self.encoder = TemporalSCLEncoder(input_dim, encoder_hidden_dim, embedding_dim, dropout)
        self.temporal_net = TemporalSCLTemporalNet(embedding_dim, temporal_hidden_dim, dropout)
        self.predictor = TemporalSCLPredictor(embedding_dim, predictor_hidden_dim, num_classes, dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            embeddings (Tensor): Normalized embeddings for each time step, shape (batch_size, seq_len, embedding_dim)
            pred_next_embeddings (Tensor): Predicted embeddings for next time steps (for time steps 0 to seq_len-2),
                                             shape (batch_size, seq_len - 1, embedding_dim)
            true_next_embeddings (Tensor): Actual embeddings for time steps 1 to seq_len,
                                             shape (batch_size, seq_len - 1, embedding_dim)
            logits (Tensor): Output logits from the predictor (e.g., using the last time step's embedding),
                             shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.size()
        # Process each time step through the encoder.
        # Flatten the time dimension to pass through the encoder.
        x_flat = x.view(-1, x.size(-1))  # (batch_size * seq_len, input_dim)
        embeddings_flat = self.encoder(x_flat)
        # Reshape back to (batch_size, seq_len, embedding_dim)
        embeddings = embeddings_flat.view(batch_size, seq_len, -1)

        # Temporal network: Predict next embeddings from the embeddings of current time steps.
        # Here, we use embeddings for time steps 0 to seq_len-2 to predict time steps 1 to seq_len.
        pred_next_embeddings = self.temporal_net(embeddings[:, :-1, :])
        true_next_embeddings = embeddings[:, 1:, :]

        # Predictor network: Generate logits using the embedding from the final time step.
        logits = self.predictor(embeddings[:, -1, :])

        return embeddings, pred_next_embeddings, true_next_embeddings, logits
