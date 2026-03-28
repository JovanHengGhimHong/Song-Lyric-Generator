import json
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.LSTM import LSTMModel
from src.utils.utils import DeviceLoader, timer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "lstm"

CONFIG = {
    "seed": 42,
    "tokenized_path": DATA_DIR / "tokenized" / "tokenized_lyrics.pt",
    "weights_path": WEIGHTS_DIR / "weights.pt",
    "config_path": WEIGHTS_DIR / "config.json",
    "batch_size": 16,
    "oom_batch_fallback": [8, 4],
    "epochs": 20,
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


# for loading the tokenized lyrics
def load_tokenized_artifact(path):
    if not path.exists():
        raise FileNotFoundError(
            "Missing tokenized artifact at "
            f"{path}. Run tokenization first to generate data/tokenized/tokenized_lyrics.pt."
        )

    return torch.load(path)


def create_loaders(artifact, batch_size, device):
    train_ds = TensorDataset(artifact["train_X"], artifact["train_Y"])
    val_ds = TensorDataset(artifact["val_X"], artifact["val_Y"])
    test_ds = TensorDataset(artifact["test_X"], artifact["test_Y"])

    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=use_pin_memory)

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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _as_float_list(values):
    return [float(v) for v in values]


# save experiment configurations and metrics for reproducibility and analysis
# data lets us plot curves also for analysis
def build_run_config(model, artifact, device, started_at, finished_at, elapsed_seconds):
    return {
        "configuration": {
            "seed": int(CONFIG["seed"]),
            "tokenized_path": str(CONFIG["tokenized_path"]),
            "weights_path": str(CONFIG["weights_path"]),
            "config_path": str(CONFIG["config_path"]),
            "batch_size": int(CONFIG["batch_size"]),
            "oom_batch_fallback": [int(v) for v in CONFIG["oom_batch_fallback"]],
            "epochs": int(CONFIG["epochs"]),
            "learning_rate": float(CONFIG["learning_rate"]),
            "input_dim": int(CONFIG["input_dim"]),
            "hidden_dim": int(CONFIG["hidden_dim"]),
            "dropout_rate": float(CONFIG["dropout_rate"]),
            "patience": int(CONFIG["patience"]),
            "optimizer": "Adam",
            "early_stop": True,
        },
        "data": {
            "vocab_size": int(artifact["vocab_size"]),
            "pad_token_id": int(artifact["pad_token_id"]),
            "sections": artifact.get("sections", []),
        },
        "metrics": {
            "loss_history": _as_float_list(model.loss_history),
            "perplexity_history": _as_float_list(model.perplexity_history),
            "val_loss_history": _as_float_list(model.val_loss_history),
            "val_perplexity_history": _as_float_list(model.val_perplexity_history),
        },
        "run_info": {
            "device": str(device),
            "actual_epochs_completed": len(model.loss_history),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(float(elapsed_seconds), 4),
        },
    }


def save_run_config(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_with_batch_fallback(artifact, device, vocab_size, pad_token_id):
    batch_candidates = [CONFIG["batch_size"], *CONFIG["oom_batch_fallback"]]
    seen = set()
    batch_candidates = [b for b in batch_candidates if b > 0 and not (b in seen or seen.add(b))]
    last_exception = None

    for batch_size in batch_candidates:
        print(f"Training attempt with batch_size={batch_size}")
        model = build_model(vocab_size, CONFIG, device)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        train_loader, val_loader, _ = create_loaders(artifact, batch_size, device)

        try:
            model.fit(
                epochs=CONFIG["epochs"],
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                early_stop=True,
                patience=CONFIG["patience"],
            )
            return model, batch_size
        except torch.OutOfMemoryError as err:
            last_exception = err
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"OOM at batch_size={batch_size}. Trying a smaller batch.")

    raise RuntimeError(
        "All configured batch sizes failed with CUDA OOM. "
        f"Tried {batch_candidates}. Last error: {last_exception}"
    )


def main():
    set_seed(CONFIG["seed"])
    device = get_device()
    print(f"Using device: {device}")

    artifact = load_tokenized_artifact(CONFIG["tokenized_path"])
    print(f"Loaded tokenized artifact: {CONFIG['tokenized_path']}")

    vocab_size = int(artifact["vocab_size"])
    pad_token_id = int(artifact["pad_token_id"])

    started_at = datetime.now()
    start_perf = time.perf_counter()
    print(f"Started Time: {started_at.isoformat()}")

    with timer("Training runtime"):
        model, effective_batch_size = train_with_batch_fallback(
            artifact=artifact,
            device=device,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
        )

    finished_at = datetime.now()
    elapsed_seconds = time.perf_counter() - start_perf

    save_weights(model, CONFIG["weights_path"])
    run_config = build_run_config(
        model=model,
        artifact=artifact,
        device=device,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
    )
    run_config["run_info"]["effective_batch_size"] = int(effective_batch_size)
    save_run_config(CONFIG["config_path"], run_config)
    print(f"Saved weights to: {CONFIG['weights_path']}")
    print(f"Saved run config to: {CONFIG['config_path']}")


if __name__ == "__main__":
    main()
