import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from src.dataset import load_market_tensor, sample_random_batch_element
from src.model import Model
from src.engine import simulate_limit_execution

# --- Hyperparameters ---
BATCH_SIZE = 32
N_BATCHES = 256
N_EPOCHS = 10
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
  print("Loading datasets into memory...")
  ohlc = load_market_tensor().to(DEVICE)
  
  model = Model(d_input=3, d_hidden=30, d_output=2).to(DEVICE)
  optimizer = optim.NAdam(model.parameters(), lr=LR)
  
  print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
  print(f"Training on device: {DEVICE}\n")

  # Setup directories for saving artifacts
  results_dir = Path("results")
  results_dir.mkdir(exist_ok=True)
  
  # Tracking metrics for plotting
  history_pnl = []
  history_winrate = []

  for epoch in range(1, N_EPOCHS + 1):
    model.train()
    total_gain, total_wins = 0.0, 0.0
    pbar = tqdm(range(1, N_BATCHES + 1), desc=f"Epoch {epoch}/{N_EPOCHS}")
    
    for n in pbar:
      optimizer.zero_grad()
      pnls = []
      
      for _ in range(BATCH_SIZE):
        x, sigma, t0, asset, patience = sample_random_batch_element(ohlc)
        x = x.to(DEVICE)
        sigma = sigma.to(DEVICE)
        
        base_price = ohlc[t0, asset, 3]
        
        # Model predicts offset multipliers
        preds = model((x / sigma).unsqueeze(0)).squeeze(0)
        bprice, sprice = (preds * sigma).exp() * base_price
        
        pnl = simulate_limit_execution(
          ohlc_history=ohlc,
          bprice=bprice,
          sprice=sprice,
          t0=t0,
          asset=asset,
          patience=patience
        )
        pnls.append(pnl)

      pnls_tensor = torch.stack(pnls)
      gain = pnls_tensor.mean()
      
      # Maximize PnL -> Minimize -PnL
      (-gain).backward()
      optimizer.step()

      total_gain += gain.item()
      total_wins += (pnls_tensor > 0.0).float().mean().item()
      
      avg_pnl = total_gain / n
      win_rate = total_wins / n
      pbar.set_postfix_str(f"avg_pnl={avg_pnl:.4f}, win_rate={win_rate:.4f}")
    
    # Store epoch results
    history_pnl.append(avg_pnl)
    history_winrate.append(win_rate)

  # --- Post-Training: Save Artifacts ---
  print("\nTraining completed. Saving artifacts...")
  
  # 1. Save model weights
  model_path = results_dir / "quant_gru_weights.pt"
  torch.save(model.state_dict(), model_path)
  
  # 2. Plot and save training curves
  plt.figure(figsize=(12, 5))
  
  plt.subplot(1, 2, 1)
  plt.plot(range(1, N_EPOCHS + 1), history_pnl, marker='o', color='red')
  plt.title("Average PnL per Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("PnL")
  plt.grid(True, linestyle='--', alpha=0.6)
  
  plt.subplot(1, 2, 2)
  plt.plot(range(1, N_EPOCHS + 1), history_winrate, marker='o', color='blue')
  plt.title("Win Rate per Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Win Rate")
  plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5) # 50% baseline
  plt.grid(True, linestyle='--', alpha=0.6)
  
  plot_path = results_dir / "training_metrics.png"
  plt.tight_layout()
  plt.savefig(plot_path)
  plt.close()
  
  print(f"Artifacts saved to '{results_dir}/'")

if __name__ == "__main__":
  train()