import torch

def simulate_limit_execution(
  ohlc_history: torch.Tensor,
  bprice: torch.Tensor,
  sprice: torch.Tensor,
  t0: int,
  asset: int,
  patience: int,
  deposit: float = 100.0,
  maker_fee: float = 0.0002,
  taker_fee: float = 0.0005
) -> torch.Tensor:
  """
  Differentiable limit order execution engine.
  Bypasses non-differentiable threshold conditions using .detach().
  """
  n_obs = ohlc_history.shape[0]
  base_price = ohlc_history[t0, asset, 3] # Close price at t0
  qty = deposit / base_price
  
  spent = torch.tensor(0.0, device=bprice.device)
  taken = torch.tensor(0.0, device=sprice.device)
  
  spent_flag = False
  taken_flag = False

  for t in range(t0 + 1, min(t0 + patience + 2, n_obs)):
    span = t - t0 - 1
    o, h, l, c = ohlc_history[t, asset]
    
    # Patience expired: take market exit (Taker fee)
    if span == patience:
      if not spent_flag:
        spent = (bprice.detach() / bprice) * o * (1.0 + taker_fee) * qty
        spent_flag = True
      if not taken_flag:
        taken = (sprice.detach() / sprice) * o * (1.0 - taker_fee) * qty
        taken_flag = True
    else:
      # Check Buy limit fill
      if not spent_flag and l <= bprice:
        if span == 0 and o <= bprice:
          spent = (bprice / bprice.detach()) * o * (1.0 + taker_fee) * qty
        else:
          spent = bprice * (1.0 + maker_fee) * qty
        spent_flag = True

      # Check Sell limit fill
      if not taken_flag and h >= sprice:
        if span == 0 and o >= sprice:
          taken = (sprice / sprice.detach()) * o * (1.0 - taker_fee) * qty
        else:
          taken = sprice * (1.0 - maker_fee) * qty
        taken_flag = True

    if spent_flag and taken_flag:
      return taken - spent

  return taken - spent