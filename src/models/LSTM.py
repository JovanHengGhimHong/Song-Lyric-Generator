import torch
import torch.nn as nn
from .model import Model


class LSTMModel(Model):
	def __init__(self, vocab_size, input_dim, hidden_dim, output_dim, dropout_rate):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, input_dim)
		self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
		self.dropout = nn.Dropout(dropout_rate)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		embedded = self.embedding(x)
		lstm_out, _ = self.lstm(embedded)
		dropped = self.dropout(lstm_out)
		return self.fc(dropped)