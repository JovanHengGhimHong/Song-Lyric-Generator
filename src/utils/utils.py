import time
from contextlib import contextmanager


@contextmanager
def timer(label):
  start = time.perf_counter()
  
  try:
    yield
  finally:
    end = time.perf_counter()
    
    print(f"{label}: {end - start:.4f}s")


# move tensor datasets to device through custom loader loop
def move_batch_to_device(batch, device):
  x_batch, y_batch = batch
  return x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

class DeviceLoader:
  def __init__(self, data_loader, device):
    self.data_loader = data_loader
    self.device = device

  def __iter__(self):
    for batch in self.data_loader:
      yield move_batch_to_device(batch, self.device)

  def __len__(self):
    return len(self.data_loader)
    
if __name__ == '__main__':
  pass