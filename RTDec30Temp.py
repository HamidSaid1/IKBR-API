import asyncio
import pandas as pd
import numpy as np
from scipy import stats
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ib_insync import IB, Stock, LimitOrder
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize IB connection
ib = IB()

# Commission Structure
MIN_COMMISSION = 0.00
PER_SHARE_COMMISSION = 0.0035
MAX_COMMISSION_PERCENT = 0.01

class EquityManager:
    """Manages equity chunks for trading."""
    def __init__(self, total_equity):
        self.total_equity = total_equity
        self.chunk_a = self.chunk_b = total_equity / 2
        self.chunk_a_in_use = self.chunk_b_in_use = False

    def get_available_chunk(self):
        """Return an available chunk of equity."""
        self._recalibrate()
        if not self.chunk_a_in_use:
            self.chunk_a_in_use = True
            return self.chunk_a
        if not self.chunk_b_in_use:
            self.chunk_b_in_use = True
            return self.chunk_b
        return None

    def release_chunk(self, amount):
        """Release and recalibrate equity after use."""
        if self.chunk_a_in_use:
            self.chunk_a = amount
            self.chunk_a_in_use = False
        elif self.chunk_b_in_use:
            self.chunk_b = amount
            self.chunk_b_in_use = False
        self.total_equity = self.chunk_a + self.chunk_b
        self._recalibrate()

    def _recalibrate(self):
        """Balance the chunks if not in use."""
        target = self.total_equity / 2
        if not self.chunk_a_in_use and not self.chunk_b_in_use:
            self.chunk_a = self.chunk_b = target
        elif self.chunk_a_in_use:
            self.chunk_b = target
        elif self.chunk_b_in_use:
            self.chunk_a = target

def calculate_indicators(df):
    """Add indicators (RSI, Bollinger Bands, ATR, Stochastic) to the dataframe."""
    rsi = RSIIndicator(df['close'], window=14).rsi()
    bollinger = BollingerBands(df['close'], window=20, window_dev=2)
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=20).average_true_range()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=20, smooth_window=8)

    df['rsi'] = rsi
    df['bb_high'], df['bb_low'], df['bb_mid'] = bollinger.bollinger_hband(), bollinger.bollinger_lband(), bollinger.bollinger_mavg()
    df['percent_b'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    df['atr'] = atr
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Calculate Bollinger Band slope
    df['bb_slope'] = df['bb_mid'].diff() / df['bb_mid'].shift(1)

    # Calculate RSI slope
    df['rsi_slope'] = df['rsi'].diff() / df['rsi'].shift(1)

    return df

def place_limit_order(action, symbol, quantity, limit_price):
    """Place a limit order."""
    contract = Stock(symbol, 'SMART', 'USD')
    order = LimitOrder(action=action, totalQuantity=quantity, lmtPrice=limit_price)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    logging.info(f"Order placed: {action} {quantity} {symbol} at {limit_price}")
    return trade

async def get_52_week_high_low(contract):
    """Get the 52-week high and low for a given stock."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime=end_date,
        durationStr='1 Y',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True
    )
    if bars:
        high52 = max(bar.high for bar in bars)
        low52 = min(bar.low for bar in bars)
        return high52, low52
    return None, None

def calculate_tiers_and_levels(price, atr):
    """Calculate profit tiers and Fibonacci levels."""
    profit_tiers = [price + (2.5 * atr * (i + 1)) for i in range(3)]
    fibonacci_levels = [(tier_low + 0.5 * (tier_high - tier_low),
                         tier_low + 0.8 * (tier_high - tier_low))
                        for tier_low, tier_high in zip(profit_tiers[:-1], profit_tiers[1:])]
    return profit_tiers, fibonacci_levels

def calculate_regression(series, window=20):
    """Calculate linear regression slope for a given series."""
    y = series.values
    x = np.arange(len(y))
    slope = np.nan
    if len(y) >= window:
        y = y[-window:]
        x = x[-window:]
        slope, _, _, _, _ = stats.linregress(x, y)
    return slope


async def process_stock(contract, equity_manager, stop_loss_percent):
    """Process a single stock for trading in real-time."""
    total_trades = total_profit = 0
    position = 0
    buy_price = 0
    profit_tiers, fibonacci_levels = [], []

    # Get 52-week high and low
    high52, low52 = await get_52_week_high_low(contract)
    if high52 is None or low52 is None:
        logging.info(f"Unable to get 52-week high/low for {contract.symbol}. Skipping.")
        return total_trades, total_profit

    if high52 > 7 * low52:
        logging.info(f"{contract.symbol} does not meet 52-week high/low criteria. Skipping.")
        return total_trades, total_profit

    while True:  # Continuously check market conditions and execute trades
        ticker = ib.reqMktData(contract)
        ib.sleep(2)  # Fetch updated market data
        current_price = ticker.last

        if current_price is None:
            logging.warning(f"No valid market data for {contract.symbol}")
            continue

        # Calculate indicators and regressions
        bars = await ib.reqHistoricalDataAsync(contract, endDateTime='', durationStr='20 D', barSizeSetting='1 day', whatToShow='TRADES', useRTH=True)
        df = pd.DataFrame(bars)
        df = calculate_indicators(df)
        
        rsi_slope = calculate_regression(df['rsi'])
        bb_slope = calculate_regression(df['bb_mid'])
        price_slope = calculate_regression(df['close'])

        current_rsi = df['rsi'].iloc[-1]
        current_percent_b = df['percent_b'].iloc[-1]
        current_stoch_k = df['stoch_k'].iloc[-1]
        current_stoch_d = df['stoch_d'].iloc[-1]
        current_atr = df['atr'].iloc[-1]

        if position == 0:
            # Logic for opening a position
            current_chunk = equity_manager.get_available_chunk()
            if current_chunk:
                # Entry conditions
                rsi_condition = current_rsi < 30 and rsi_slope > 0
                bb_condition = current_percent_b < 0.1 and bb_slope > 0
                stoch_condition = current_stoch_k < 20 and current_stoch_k > current_stoch_d
                price_condition = price_slope > 0

                if rsi_condition and bb_condition and stoch_condition and price_condition:
                    position = int(current_chunk // current_price)
                    buy_price = current_price
                    
                    profit_tiers, fibonacci_levels = calculate_tiers_and_levels(buy_price, current_atr)
                    
                    place_limit_order('BUY', contract.symbol, position, buy_price * 0.99)
                    total_trades += 1
                    logging.info(f"Opened position for {contract.symbol} at {buy_price}.")
        else:
            # Logic for managing an open position
            if current_price <= buy_price * (1 - stop_loss_percent):
                logging.info(f"Stop-loss triggered for {contract.symbol}. Selling position.")
                total_profit += (current_price - buy_price) * position
                equity_manager.release_chunk(position * current_price)
                position = 0
            else:
                for tier, (mid_point, fib_level) in zip(profit_tiers[1:], fibonacci_levels):
                    if current_price >= mid_point:
                        logging.info(f"Profit target reached for {contract.symbol}. Selling position.")
                        total_profit += (current_price - buy_price) * position
                        equity_manager.release_chunk(position * current_price)
                        position = 0
                        break

        ib.sleep(5)  # Wait before checking again

    return total_trades, total_profit

async def real_time_trading(ticker_symbols, equity, stop_loss_percent):
    """Main real-time trading loop."""
    contracts = [Stock(symbol, 'SMART', 'USD') for symbol in ticker_symbols]
    equity_manager = EquityManager(equity)

    tasks = [process_stock(contract, equity_manager, stop_loss_percent) for contract in contracts]
    results = await asyncio.gather(*tasks)
    return results

class TradingBotGUI:
    """GUI for configuring the trading bot."""
    def __init__(self, master):
        self.master = master
        master.title("Real-Time Trading Bot")
        master.geometry("400x400")
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.master, text="Ticker Symbols (comma-separated):").pack(pady=5)
        self.ticker_entry = ttk.Entry(self.master, width=40)
        self.ticker_entry.pack(pady=5)

        ttk.Label(self.master, text="Equity to Invest:").pack(pady=5)
        self.equity_entry = ttk.Entry(self.master)
        self.equity_entry.pack(pady=5)

        ttk.Label(self.master, text="Stop Loss %:").pack(pady=5)
        self.stop_loss_entry = ttk.Entry(self.master)
        self.stop_loss_entry.insert(0, "0.02")
        self.stop_loss_entry.pack(pady=5)

        self.run_button = ttk.Button(self.master, text="Run Trading Bot", command=self.run_trading_bot)
        self.run_button.pack(pady=20)

    def run_trading_bot(self):
        try:
            symbols = [s.strip() for s in self.ticker_entry.get().split(',')]
            equity = float(self.equity_entry.get())
            stop_loss_percent = float(self.stop_loss_entry.get())
            asyncio.run(real_time_trading(symbols, equity, stop_loss_percent))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
        root = tk.Tk()
        app = TradingBotGUI(root)
        root.mainloop()
    finally:
        ib.disconnect()