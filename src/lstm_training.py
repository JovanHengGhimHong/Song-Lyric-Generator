import os
import random

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.LSTM import LSTMModel
from src.utils.utils import DeviceLoader, timer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
	"seed": 42,
	"tokenized_path": os.path.join(ROOT_DIR, "data", "tokenized", "tokenized_lyrics.pt"),
	"weights_path": os.path.join(ROOT_DIR, "weights", "lstm_model.pt"),
	"batch_size": 16,
	"epochs": 30,
	"learning_rate": 1e-3,
	"input_dim": 128,
	"hidden_dim": 256,
	"dropout_rate": 0.3,
	"patience": 3,
}


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def get_device():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenized_artifact(path):
	if not os.path.exists(path):
		raise FileNotFoundError(
			"Missing tokenized artifact at "
			f"{path}. Run tokenization first to generate data/tokenized/tokenized_lyrics.pt."
		)

	return torch.load(path)


def create_loaders(artifact, batch_size, device):
	train_ds = TensorDataset(artifact["train_X"], artifact["train_Y"])
	val_ds = TensorDataset(artifact["val_X"], artifact["val_Y"])
	test_ds = TensorDataset(artifact["test_X"], artifact["test_Y"])

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

	return DeviceLoader(train_loader, device), DeviceLoader(val_loader, device), DeviceLoader(test_loader, device)


def build_model(vocab_size, config, device):
	model = LSTMModel(
		vocab_size=vocab_size,
		input_dim=config["input_dim"],
		hidden_dim=config["hidden_dim"],
		output_dim=vocab_size,
		dropout_rate=config["dropout_rate"],
	)
	return model.to(device)


def save_weights(model, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(model.state_dict(), path)


def main():
	set_seed(CONFIG["seed"])
	device = get_device()
	print(f"Using device: {device}")

	artifact = load_tokenized_artifact(CONFIG["tokenized_path"])
	print(f"Loaded tokenized artifact: {CONFIG['tokenized_path']}")

	train_loader, val_loader, _ = create_loaders(artifact, CONFIG["batch_size"], device)

	vocab_size = int(artifact["vocab_size"])
	pad_token_id = int(artifact["pad_token_id"])

	model = build_model(vocab_size, CONFIG, device)
	loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
	optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

	with timer("Training runtime"):
		model.fit(
			epochs=CONFIG["epochs"],
			train_loader=train_loader,
			val_loader=val_loader,
			loss_fn=loss_fn,
			optimizer=optimizer,
			early_stop=True,
			patience=CONFIG["patience"],
		)

	save_weights(model, CONFIG["weights_path"])
	print(f"Saved weights to: {CONFIG['weights_path']}")


if __name__ == "__main__":
	main()
