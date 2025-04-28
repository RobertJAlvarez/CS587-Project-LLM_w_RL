import torch  # type: ignore
import torch.nn as nn  # type: ignore


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, 1)

    def forward(self, token_embeddings):
        # token_embeddings: (batch_size, seq_len, embed_dim)
        attn_weights = torch.softmax(self.query(token_embeddings), dim=1)
        return (attn_weights * token_embeddings).sum(dim=1)  # (batch_size, embed_dim)


class SamplingPolicy(nn.Module):
    """Learn temperature and top-p adjustments based on prompt embeddings."""

    def __init__(self, embed_dim=256) -> None:
        super().__init__()
        self.attention_pooling = AttentionPooling(embed_dim)

        # Output 2 values: [temperature, top_p]
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 2)
        )

    def forward(self, token_embeddings) -> tuple[torch.Tensor, torch.Tensor]:
        # outputs.shape = (batch_size, embed_dim)
        outputs = self.mlp(self.attention_pooling(token_embeddings))
        temperature = torch.sigmoid(outputs[:, 0]) * 2.0  # (0, 2)
        top_p = 0.8 + torch.sigmoid(outputs[:, 1]) * 0.2  # (0.8, 1.0)
        return temperature, top_p
