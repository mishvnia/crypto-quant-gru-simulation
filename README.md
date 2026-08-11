# Quantitative Market Strategy Research (PyTorch GRU)

An R&D quantitative research project focused on testing market predictability and limit-order execution on Binance Perpetual Futures. The project features stochastic dataset sampling, custom PyTorch neural architecture, and a fully differentiable simulation of bid/ask limit-order execution.

## 📊 Dataset
The dataset comprises 5 years (2021-2026) of 1-minute OHLCV data for BTC, ETH, SOL, and BNB collected via Binance API.
**Link to Dataset:** [Binance 1-Minute Perpetual Futures OHLCV 2021-2026](https://www.kaggle.com/datasets/mishvnia/binance-1m-perpetual-futures-ohlcv-2021-2026)

## 🛠 Tech Stack
- **Deep Learning:** PyTorch (`nn.GRU`, custom loss backpropagation)
- **Data Engineering & Wrangling:** Pandas, NumPy, Binance Spot & UM Futures API
- **Visualization & Utilities:** Matplotlib, tqdm

## 🧠 Methodology & Strategy Design

1. **Trading Strategy:**
   The model attempts to set simultaneous **Buy (Bid)** and **Sell (Ask)** limit orders around the current price based on predicted volatility and return expectations. The execution engine accounts for order patience, maker/taker fees, and high/low candle overlap.

2. **Full Stochastic Sampling (Absolute Shuffling):**
   To prevent overfitting to specific market regimes or single-asset dynamics, the training loop applies absolute random sampling for every batch element:
   - **Random Asset:** Randomly selected from available futures contracts.
   - **Random Timeframe (Resampling):** Dynamically aggregated candles between 1 minute and 8 hours.
   - **Random Context Window:** Sequence history varies dynamically in length (`n_frames`).

3. **Differentiable PnL Engine:**
   Implemented a trick using `detach()` in PyTorch to bypass non-differentiable limit-order trigger conditions (whether high/low touches the price). This allows the neural network to backpropagate gradients directly through the simulated PnL.

4. **Normalization:**
   Features are dynamically scaled by the rolling standard deviation ($\sigma$) of logarithmic returns to make inputs invariant to historical price scales.

## 📈 Key Findings & Fractal Noise Conclusion

* **Results:** Average PnL remains negative with a Win Rate around 44%.
* **Conclusion (Fractal Noise Hypothesis):**
  Despite dynamic timeframe resampling and heavy cross-asset shuffling, the model fails to extract a profitable edge. This demonstrates that the **fractal nature of financial markets effectively scales the underlying noise across all timeframes**. The self-similarity of market dynamics means that strategy execution on purely technical/OHLC features faces the same efficiency barrier whether evaluated on 1-minute or multi-hour windows once transaction fees are introduced.
