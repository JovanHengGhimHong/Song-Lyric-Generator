from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PARQUET_PATH = DATA_DIR / "pop_songs.parquet"
TOKENIZER_OUTPUT_DIR = DATA_DIR / "tokenizer"
TOKENIZED_OUTPUT_DIR = DATA_DIR / "tokenized"
TOKENIZED_OUTPUT_PATH = TOKENIZED_OUTPUT_DIR / "tokenized_lyrics.pt"

SECTIONS = ["VERSE", "CHORUS", "BRIDGE", "INTRO", "OUTRO", "PRE-CHORUS", "HOOK"]

SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(
			f"Missing input data at {path}. Run ETL first to generate data/pop_songs.parquet."
		)

	df = pd.read_parquet(path)
	if "lyrics" not in df.columns:
		raise ValueError("Expected a 'lyrics' column in the dataset.")

	return df


def split_dataset(df: pd.DataFrame, seed: int):
	train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=seed)
	train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=seed)
	return train_df, val_df, test_df


def build_tokenizers():
	tokenizer_train = BertTokenizer.from_pretrained("bert-base-uncased")
	tokenizer_val = BertTokenizer.from_pretrained("bert-base-uncased")
	tokenizer_test = BertTokenizer.from_pretrained("bert-base-uncased")

	section_tokens = [f"<{section}>" for section in SECTIONS]
	tokenizer_train.add_tokens(section_tokens)
	tokenizer_val.add_tokens(section_tokens)
	tokenizer_test.add_tokens(section_tokens)

	return tokenizer_train, tokenizer_val, tokenizer_test


def tokenize_splits(train_df, val_df, test_df, tokenizer_train, tokenizer_val, tokenizer_test):
	train_X = tokenizer_train(
		train_df["lyrics"].tolist(),
		padding=True,
		truncation=True,
		return_tensors="pt",
	)["input_ids"]
	val_X = tokenizer_val(
		val_df["lyrics"].tolist(),
		padding=True,
		truncation=True,
		return_tensors="pt",
	)["input_ids"]
	test_X = tokenizer_test(
		test_df["lyrics"].tolist(),
		padding=True,
		truncation=True,
		return_tensors="pt",
	)["input_ids"]

	return train_X, val_X, test_X


def create_shifted_labels(x_tensor: torch.Tensor, pad_token_id: int) -> torch.Tensor:
	y_tensor = torch.zeros_like(x_tensor)
	y_tensor[:, :-1] = x_tensor[:, 1:]
	y_tensor[:, -1] = pad_token_id
	return y_tensor


def save_artifacts(train_X, train_Y, val_X, val_Y, test_X, test_Y, vocab_size, pad_token_id, tokenizer_train):
	TOKENIZED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	TOKENIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	torch.save(
		{
			"train_X": train_X,
			"train_Y": train_Y,
			"val_X": val_X,
			"val_Y": val_Y,
			"test_X": test_X,
			"test_Y": test_Y,
			"vocab_size": vocab_size,
			"pad_token_id": pad_token_id,
			"sections": SECTIONS,
		},
		TOKENIZED_OUTPUT_PATH,
	)

	tokenizer_train.save_pretrained(TOKENIZER_OUTPUT_DIR)


def main():
	set_seed(SEED)

	print(f"Loading data from: {INPUT_PARQUET_PATH}")
	df = load_dataset(INPUT_PARQUET_PATH)

	train_df, val_df, test_df = split_dataset(df, seed=SEED)
	print(f"Split sizes | train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

	tokenizer_train, tokenizer_val, tokenizer_test = build_tokenizers()
	train_X, val_X, test_X = tokenize_splits(
		train_df,
		val_df,
		test_df,
		tokenizer_train,
		tokenizer_val,
		tokenizer_test,
	)

	pad_token_id = tokenizer_train.pad_token_id
	vocab_size = len(tokenizer_train)

	train_Y = create_shifted_labels(train_X, pad_token_id)
	val_Y = create_shifted_labels(val_X, pad_token_id)
	test_Y = create_shifted_labels(test_X, pad_token_id)

	save_artifacts(
		train_X,
		train_Y,
		val_X,
		val_Y,
		test_X,
		test_Y,
		vocab_size,
		pad_token_id,
		tokenizer_train,
	)

	print(f"Saved tokenized outputs to: {TOKENIZED_OUTPUT_PATH}")
	print(f"Saved tokenizer artifacts to: {TOKENIZER_OUTPUT_DIR}")
	print(f"vocab_size={vocab_size}, pad_token_id={pad_token_id}")


if __name__ == "__main__":
	main()
