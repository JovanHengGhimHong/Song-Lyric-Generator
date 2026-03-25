import torch
import numpy as np
import torch.nn as nn

class Model(nn.Module):
  '''
  This Model serves as a base class for all models.
  
  This model predicts on recall
  '''
  def __init__(self):
    super().__init__()
    self.accuracy_history = []
    self.loss_history = []
    # user defines layers 
   
  def forward(self, x):
    # user defines forward pass
    return x

  def fit(self, epochs, train_loader, loss_fn, optimizer, early_stop=False, patience=5):
    super().train()

    if patience < 1:
      raise ValueError("Patience must be at least 1.")

    if early_stop is False and patience > 0:
      print("Warning: Patience is set but early_stop is False. Patience will be ignored.")

    min_loss = np.inf
    wait = 0
    for i in range(epochs):
      epoch_loss = 0.0
      total = 0 
      correct = 0 
      for x_batch, y_batch in train_loader:
        # reset gradients
        optimizer.zero_grad()

        logits = self(x_batch)

        # back propagation
        loss = loss_fn(logits, y_batch)
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # store weights
        epoch_loss += loss.item()
        
        # compute accuracy
        predictions = logits.argmax(dim=1)
      
        total += y_batch.size(0)
        correct += (predictions == y_batch).sum().item()
        

      epoch_accuracy = correct / total * 100

      # storing
      self.accuracy_history.append(epoch_accuracy)
      self.loss_history.append(epoch_loss)

      print(f"Epoch {i+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

      # early stopping check
      # early stop to always track loss
      if early_stop == False:
        continue
      
      if epoch_loss < min_loss:
        min_loss = epoch_loss
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
        preds = logits.argmax(dim=1)
        predictions.extend(preds.tolist())

    return predictions

    
if __name__ == '__main__':
  pass