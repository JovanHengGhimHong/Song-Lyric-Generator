import torch
import numpy as np
import torch.nn as nn

def perplexity_from_loss(loss_value):
  return float(np.exp(loss_value))

class Model(nn.Module):
  '''
  This Model serves as a base class for all models.
  
  This model predicts on recall
  '''
  def __init__(self):
    super().__init__()
    self.loss_history = []
    self.perplexity_history = []
    self.val_loss_history = []
    self.val_perplexity_history = []
    # user defines layers 
   
  def forward(self, x):
    # user defines forward pass
    return x

  def _flatten_outputs(self, logits, targets):
    # Handles token-level logits from language models and class logits from classifiers.
    if logits.dim() == 3:
      return logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    return logits, targets

  def fit(
    self,
    epochs,
    train_loader,
    loss_fn,
    optimizer,
    val_loader=None,
    early_stop=False,
    patience=5
  ):
    super().train()

    if patience < 1:
      raise ValueError("Patience must be at least 1.")

    if early_stop is False and patience > 0:
      print("Warning: Patience is set but early_stop is False. Patience will be ignored.")

    min_loss = np.inf
    wait = 0

    for i in range(epochs):
      self.train()
      epoch_loss = 0.0

      for x_batch, y_batch in train_loader:
        # reset gradients
        optimizer.zero_grad()

        logits = self(x_batch)
        flat_logits, flat_targets = self._flatten_outputs(logits, y_batch)

        # back propagation
        loss = loss_fn(flat_logits, flat_targets)
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # store loss
        epoch_loss += loss.item()

      epoch_loss = epoch_loss / len(train_loader)
      epoch_perplexity = perplexity_from_loss(epoch_loss)

      # storing
      self.loss_history.append(epoch_loss)
      self.perplexity_history.append(epoch_perplexity)

      val_loss = None
      val_perplexity = None
      if val_loader is not None:
        self.eval()
        running_val_loss = 0.0
        with torch.no_grad():
          for x_val, y_val in val_loader:
            val_logits = self(x_val)
            flat_val_logits, flat_val_targets = self._flatten_outputs(val_logits, y_val)
            batch_val_loss = loss_fn(flat_val_logits, flat_val_targets)
            running_val_loss += batch_val_loss.item()

        val_loss = running_val_loss / len(val_loader)
        val_perplexity = perplexity_from_loss(val_loss)
        self.val_loss_history.append(val_loss)
        self.val_perplexity_history.append(val_perplexity)

      if val_loss is not None:
        print(
          f"Epoch {i+1}/{epochs} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Perplexity: {epoch_perplexity:.2f} | "
          f"Val Perplexity: {val_perplexity:.2f}"
        )
      else:
        print(
          f"Epoch {i+1}/{epochs} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Perplexity: {epoch_perplexity:.2f}"
        )

      # early stopping check
      if early_stop is False:
        continue

      monitored_loss = val_loss if val_loss is not None else epoch_loss
      if monitored_loss < min_loss:
        min_loss = monitored_loss
        wait = 0
        continue

      wait += 1
      if wait >= patience:
        print(f"Early stopping at epoch {i+1} due to no improvement in validation loss for {patience} consecutive epochs.")
        break

  def predict(self, test_loader):
    super().eval()
    predictions = []

    with torch.no_grad():
      for x_batch, _ in test_loader:
        logits = self(x_batch)
        if logits.dim() == 3:
          preds = logits.argmax(dim=-1)
        else:
          preds = logits.argmax(dim=1)
        predictions.extend(preds.tolist())

    return predictions

    
if __name__ == '__main__':
  pass