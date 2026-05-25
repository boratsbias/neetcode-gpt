import torch
import torch.nn as nn
from torchtyping import TensorType

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.get_keys = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.get_queries = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.get_values = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Return your answer to 4 decimal places
        k = self.get_keys(embedded)
        q = self.get_queries(embedded)
        v = self.get_values(embedded)

        scores = q @ torch.transpose(k, 1, 2)
        B, T, A = k.shape
        scores = scores / (A ** 0.5)

        pre_mask = torch.tril(torch.ones(T, T))
        mask = pre_mask == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim=2)
        transformed = scores @ v
        return torch.round(transformed, decimals=4)






