import urllib.request
import zipfile
import duckdb
import socket
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Set global timeout for socket connections (prevents infinite hanging)
socket.setdefaulttimeout(30)

# --- Configuration ---
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
START_DATE = datetime(2021, 7, 1)
END_DATE = datetime(2026, 6, 1) # Inclusive up to June 2026
INTERVAL = '1m'
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

# Directories
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# DuckDB datatypes for Binance Futures UM CSV
DTYPES_DICT = {
  'column00': 'BIGINT', 'column01': 'DECIMAL(19, 8)', 'column02': 'DECIMAL(19, 8)',
  'column03': 'DECIMAL(19, 8)', 'column04': 'DECIMAL(19, 8)', 'column05': 'DECIMAL(19, 8)',
  'column06': 'BIGINT', 'column07': 'DECIMAL(19, 8)', 'column08': 'BIGINT',
  'column09': 'DECIMAL(19, 8)', 'column10': 'DECIMAL(19, 8)', 'column11': 'VARCHAR'
}

def get_months_list(start: datetime, end: datetime) -> list:
  """Generate a list of YYYY-MM strings between start and end dates."""
  months = []
  current = start
  while current <= end:
    months.append(current.strftime("%Y-%m"))
    current += relativedelta(months=1)
  return months

def download_and_extract(symbol: str, months: list, max_retries: int = 3):
  """Download monthly ZIP files from Binance with retries and extract CSVs."""
  symbol_dir = RAW_DIR / symbol
  symbol_dir.mkdir(exist_ok=True)
  
  print(f"\n--- Downloading data for {symbol} ---")
  for month in tqdm(months, desc=f"Fetching {symbol}"):
    zip_name = f"{symbol}-{INTERVAL}-{month}.zip"
    url = f"{BASE_URL}/{symbol}/{INTERVAL}/{zip_name}"
    zip_path = symbol_dir / zip_name
    csv_path = symbol_dir / f"{symbol}-{INTERVAL}-{month}.csv"
    
    # Download with retry mechanism
    if not zip_path.exists() and not csv_path.exists():
      for attempt in range(max_retries):
        try:
          urllib.request.urlretrieve(url, zip_path)
          break # Success, exit retry loop
        except (Exception, socket.timeout) as e:
          print(f"\nAttempt {attempt + 1}/{max_retries} failed for {zip_name}: {e}")
          # Remove potentially corrupted partial download
          if zip_path.exists():
            zip_path.unlink()
          if attempt < max_retries - 1:
            time.sleep(2) # Wait a bit before retrying
          else:
            print(f"Failed to download {zip_name} after {max_retries} attempts.")
            continue # Skip to the next month

    # Extract and remove ZIP
    if zip_path.exists():
      try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
          zip_ref.extractall(symbol_dir)
        zip_path.unlink()
      except zipfile.BadZipFile:
        print(f"\nCorrupted ZIP file found and removed: {zip_name}")
        zip_path.unlink()

def process_and_validate(symbol: str):
  """Aggregate CSVs using DuckDB, validate data integrity, and save to Parquet."""
  print(f"\n--- Processing & Validating {symbol} ---")
  symbol_dir = RAW_DIR / symbol
  parquet_path = PROCESSED_DIR / f"{symbol}_2021_2026.parquet"
  
  conn = duckdb.connect()
  
  create_view_query = f"""
    CREATE VIEW df AS
    SELECT 
      *,
      volume - buy_volume AS sell_volume,
      qav - buy_qav AS sell_qav
    FROM '{parquet_path}'
  """

  try: conn.execute(create_view_query)
  except:
    print("Compiling CSVs into Parquet...")
    conn.execute(f"""
      COPY (
        SELECT 
          epoch_ms(column00) AS open_time,
          column01 AS open,
          column02 AS high,
          column03 AS low,
          column04 AS close,
          column05 AS volume,
          column07 AS qav,
          column08 AS n_trades,
          column09 AS buy_volume,
          column10 AS buy_qav
        FROM read_csv('{symbol_dir}/*.csv', header=False, columns={DTYPES_DICT}, ignore_errors=true)
        ORDER BY open_time
      ) TO '{parquet_path}' (FORMAT PARQUET)
    """)
    conn.execute(create_view_query)
  
  print("Running integrity checks...")
  
  assert conn.execute('''
    WITH expected_index AS (
      SELECT unnest(generate_series(
        TIMESTAMP '2021-07-01 00:00:00', 
        TIMESTAMP '2026-07-01 00:00:00' - INTERVAL 1 MINUTE, 
        INTERVAL 1 MINUTE
      )) AS ts
    ),
    diff AS (
      (SELECT ts FROM expected_index EXCEPT SELECT open_time FROM df)
      UNION ALL
      (SELECT open_time FROM df EXCEPT SELECT ts FROM expected_index)
    )
    SELECT COUNT(*) = 0 FROM diff
  ''').fetchone()[0]

  assert conn.execute('''
    SELECT bool_and(
      open >= 0 AND high >= 0 AND low >= 0 AND close >= 0 AND 
      volume >= 0 AND qav >= 0 AND n_trades >= 0 AND 
      buy_volume >= 0 AND buy_qav >= 0 AND 
      sell_volume >= 0 AND sell_qav >= 0
    ) FROM df
  ''').fetchone()[0]

  assert conn.execute('''
    SELECT bool_and(
      (n_trades > 0) = (volume > 0) AND
      (qav > 0) = (volume > 0) AND
      (buy_qav > 0) = (buy_volume > 0) AND
      (sell_qav > 0) = (sell_volume > 0)
    ) FROM df
  ''').fetchone()[0]

  assert conn.execute('''
    SELECT bool_and(
      low > 0 AND
      open >= low AND open <= high AND
      close >= low AND close <= high
    ) FROM df
  ''').fetchone()[0]

  assert conn.execute('''
    WITH ffilled AS (
      SELECT 
        volume, open, high, low, close,
        last_value(CASE WHEN volume > 0 THEN close ELSE NULL END IGNORE NULLS) 
        OVER (ORDER BY open_time) AS ffill_close
      FROM df
    )
    SELECT bool_and(
      volume > 0 OR (
        ffill_close IS NOT NULL AND
        open = ffill_close AND
        high = ffill_close AND
        low = ffill_close AND
        close = ffill_close
      ) OR (
        ffill_close IS NULL AND 
        open = close AND high = close AND low = close
      )
    ) FROM ffilled
  ''').fetchone()[0]

  assert conn.execute('''
    SELECT bool_and(
      (volume = 0 OR (
        qav >= low * volume AND 
        qav <= high * volume
      )) AND
      (buy_volume = 0 OR (
        buy_qav >= low * buy_volume AND 
        buy_qav <= high * buy_volume
      )) AND
      (sell_volume = 0 OR (
        sell_qav >= low * sell_volume AND 
        sell_qav <= high * sell_volume
      ))
    ) FROM df
  ''').fetchone()[0]

  print("All checks passed successfully!")
  conn.close()

  try:
    print("Cleaning up raw CSV files...")
    for csv_file in symbol_dir.glob("*.csv"):
      csv_file.unlink()
    symbol_dir.rmdir()
  except: print("Clean up failed")

def main():
  months = get_months_list(START_DATE, END_DATE)
  
  for symbol in SYMBOLS:
    try: process_and_validate(symbol)
    except:
      download_and_extract(symbol, months)
      process_and_validate(symbol)
    
  print(f"\nPipeline completed! Parquet files are stored in {PROCESSED_DIR}")

if __name__ == "__main__":
  main()