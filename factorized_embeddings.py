import torch
import torch.nn as nn

class FactorizedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding_matrix_1 = nn.Parameter(torch.randn(num_embeddings, hidden_dim))
        self.embedding_matrix_2 = nn.Parameter(torch.randn(hidden_dim, embedding_dim))

    def forward(self, input_ids):
        return torch.matmul(self.embedding_matrix_1[input_ids], self.embedding_matrix_2.T)
    

def initialize_with_svd(original_embedding: nn.Embedding, hidden_dim: int):
    with torch.no_grad():
        weight = original_embedding.weight.data
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

    U = U[:, :hidden_dim]
    S = S[:hidden_dim]
    Vt = Vt[:hidden_dim, :]

    embedding_matrix_1 = U @ torch.diag(S)
    embedding_matrix_2 = Vt

    factorized_embedding = FactorizedEmbedding(
        num_embeddings=original_embedding.num_embeddings,
        embedding_dim=original_embedding.embedding_dim,
        hidden_dim=hidden_dim
    )
    factorized_embedding.embedding_matrix_1.data = embedding_matrix_1
    factorized_embedding.embedding_matrix_2.data = embedding_matrix_2.T
    return factorized_embedding
