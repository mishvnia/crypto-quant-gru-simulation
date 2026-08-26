import torch
from torch import nn

class Model(nn.Module):
  def __init__(self, d_input: int = 3, d_hidden: int = 30, d_output: int = 2):
    super().__init__()
    self.gru = nn.GRU(d_input, d_hidden, batch_first=True)
    self.fc = nn.Linear(d_hidden, d_output)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x shape: [batch_size, n_frames, d_input]
    # Extract last hidden state: gru(x)[1][0] -> shape: [batch_size, d_hidden]
    out, h_n = self.gru(x)
    return self.fc(h_n[0])