from pathlib import Path
import duckdb
import numpy as np
import torch

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
PROCESSED_DIR = Path("data/processed")

def load_market_tensor() -> torch.Tensor:
  """Loads all Parquet files into a single 3D PyTorch Tensor [n_obs, n_assets, 4]."""
  tensor_list = []
  conn = duckdb.connect()
  
  for symbol in SYMBOLS:
    parquet_path = PROCESSED_DIR / f"{symbol}_2021_2026.parquet"
    if not parquet_path.exists():
      raise FileNotFoundError(f"Missing {parquet_path}. Run data_pipeline.py first!")
    
    # Fetch Open, High, Low, Close
    arr = conn.execute(f"""
      SELECT open, high, low, close 
      FROM '{parquet_path}' 
      ORDER BY open_time
    """).fetchnumpy()
    
    ohlc = np.column_stack([arr['open'], arr['high'], arr['low'], arr['close']])
    tensor_list.append(torch.from_numpy(ohlc))
    
  conn.close()
  # Stack along assets dimension: [n_obs, n_assets, 4]
  return torch.stack(tensor_list, dim=1)

def sample_random_batch_element(
  ohlc: torch.Tensor,
  min_tf: int = 60,
  max_tf: int = 480,
  min_frames: int = 10,
  max_frames: int = 30
):
  """
  Samples a single stochastic regime:
  - Random asset
  - Random timeframe aggregation
  - Random lookback window (via .reshape)
  """
  n_obs, n_assets, _ = ohlc.shape
  
  asset = np.random.choice(n_assets)
  tf = np.random.randint(min_tf, max_tf + 1)
  n_frames = np.random.randint(min_frames, max_frames + 1)
  lookback = n_frames * tf
  patience = tf
  t0 = np.random.randint(lookback, n_obs - patience - 1)

  # Extract window for resampling: shape [lookback + 1, 3] (High, Low, Close)
  sample = ohlc[t0 - lookback: t0 + 1, asset, 1:]
  
  # Reshape instead of unfold
  frames = sample[1:].reshape(n_frames, tf, 3)
  
  # Base for returns is the close of previous frame: sample[:-1:tf, 2]
  prev_close = sample[:-1:tf, 2]
  
  x = torch.stack((
    frames[..., 0].max(dim=1).values / prev_close, # Normalized High
    frames[..., 1].min(dim=1).values / prev_close, # Normalized Low
    sample[tf::tf, 2] / prev_close                 # Normalized Close
  ), dim=1).log().float()
  
  sigma = x[:, 2].std()
  if sigma == 0 or torch.isnan(sigma):
    sigma = torch.tensor(1e-4)

  return x, sigma, t0, asset, patience