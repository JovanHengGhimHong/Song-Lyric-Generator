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

    
if __name__ == '__main__':
  pass