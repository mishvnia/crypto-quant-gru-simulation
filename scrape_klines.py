import pandas as pd
from tqdm import tqdm
from binance.spot import Spot
from binance.um_futures import UMFutures

def scrape_klines(index, func, *args, limit=1_000, verbose=True):

  beg = int(index[0].timestamp() * 1e3)
  end = int(index[-1].timestamp() * 1e3)
  step = index.freq.nanos // 1_000_000 * limit
  pbar = range(beg, end + step, step)
  if verbose: pbar = tqdm(pbar, 'paging')

  data = []
  for t in pbar:
    chunk = func(*args, limit=limit, startTime=t)
    data.extend(chunk)

  col_seq = pd.Index((
    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
    'Quote asset volume', 'Number of trades', 'Taker buy volume', 'Taker buy QAV', 'Ignore'
  ))
  del_cols = pd.Index(('Close time', 'Ignore'))
  idx_col = 'Open time'

  df = pd.DataFrame(data, columns=col_seq)
  df.drop(columns=del_cols, inplace=True)

  df[idx_col] = pd.to_datetime(df[idx_col], 'coerce', unit='ms')
  df.dropna(subset=idx_col, inplace=True)
  df.drop_duplicates(idx_col, inplace=True)
  df.set_index(idx_col, inplace=True)

  return df.apply(pd.to_numeric, errors='coerce').reindex(index)