    # December 24th Test: Full Backtest Script with Enhancements
    import asyncio
    import pandas as pd
    import numpy as np
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import AverageTrueRange, BollingerBands
    from ib_insync import IB, Stock, util
    import tkinter as tk
    from tkinter import simpledialog, ttk, messagebox
    import logging
    from datetime import datetime, timedelta

    # Commission Structure
    MIN_COMMISSION = 0.00
    PER_SHARE_COMMISSION = 0.0035
    MAX_COMMISSION_PERCENT = 0.01

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize IB connection
    ib = IB()

    def is_end_of_day(current_time, previous_time):
        """Determine if the current time is at the end of the trading day."""
        # Ensure both are datetime objects
        if isinstance(current_time, int):
            current_time = datetime.fromtimestamp(current_time)  # Convert timestamp to datetime
        elif isinstance(current_time, pd.Timestamp):
            current_time = current_time.to_pydatetime()
        
        if isinstance(previous_time, int):
            previous_time = datetime.fromtimestamp(previous_time)  # Convert timestamp to datetime
        elif isinstance(previous_time, pd.Timestamp):
            previous_time = previous_time.to_pydatetime()
        
        return current_time.date() != previous_time.date()



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

    def calculate_slopes(data, window):
        """Calculate slopes of the given data over the specified window."""
        slopes = []
        for i in range(len(data) - window + 1):
            y = data[i:i+window]
            x = np.arange(window)
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(slope)
        return slopes

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
        return df

    def calculate_tiers_and_levels(price, atr):
        """Calculate profit tiers and Fibonacci levels."""
        profit_tiers = [price + (2.5 * atr * (i + 1)) for i in range(3)]
        fibonacci_levels = [(tier_low + 0.5 * (tier_high - tier_low),
                            tier_low + 0.8 * (tier_high - tier_low))
                            for tier_low, tier_high in zip(profit_tiers[:-1], profit_tiers[1:])]
        return profit_tiers, fibonacci_levels

    def evaluate_market_conditions(df, i, num_data_points, adjacent_segments):
        """Evaluate market conditions to decide trading actions."""
        current_price = df['close'].iloc[i]
        current_atr = df['atr'].iloc[i]

        # Calculate raw slopes
        rsi_slopes = calculate_slopes(df['rsi'].iloc[i-num_data_points+1:i+1], adjacent_segments)
        percent_b_slopes = calculate_slopes(df['percent_b'].iloc[i-num_data_points+1:i+1], adjacent_segments)
        stoch_k_slopes = calculate_slopes(df['stoch_k'].iloc[i-num_data_points+1:i+1], adjacent_segments)

        # Adjust Stochastic Oscillator weights for better accuracy
        stoch_k_slopes = [slope * 1.25 for slope in stoch_k_slopes]

        # Calculate regression slopes
        rsi_regression = np.polyfit(range(len(rsi_slopes)), rsi_slopes, 1)[0]
        percent_b_regression = np.polyfit(range(len(percent_b_slopes)), percent_b_slopes, 1)[0]
        stoch_k_regression = np.polyfit(range(len(stoch_k_slopes)), stoch_k_slopes, 1)[0]

        # Scoring system
        score = max(0, abs(rsi_regression) + percent_b_regression + stoch_k_regression)
        return score, current_price, current_atr


    async def get_52_week_data(ib, contract):
        """Get the 52-week high, low, and range for a given stock."""
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
            range52 = high52 - low52
            return high52, low52, range52
        return None, None, None

    def calculate_nprv(current_price, high52, low52, range52, atr):
        """Calculate the Normalized Price Range Volatility (NPRV)."""
        if range52 == 0 or atr == 0:
            return None
        price_position = (current_price - low52) / range52
        return price_position / atr

    async def process_stock(contract, equity_manager, num_data_points, adjacent_segments, stop_loss_percent, df):
        """Process a single stock for trading."""
        total_trades = total_profit = position = buy_price = 0
        profit_tiers, fibonacci_levels = [], []

        # Get 52-week high, low, and range
        high52, low52, range52 = await get_52_week_data(ib, contract)
        
        if high52 is None or low52 is None or range52 is None:
            print(f"Unable to get 52-week data for {contract.symbol}. Skipping this stock.")
            return total_trades, total_profit

        # Check if the stock meets the 52-week high/low criteria
        if high52 > 12 * low52:
            print(f"{contract.symbol} does not meet the 52-week high/low criteria. Skipping this stock.")
            return total_trades, total_profit

        for i in range(num_data_points, len(df)):
            current_price, current_atr = df['close'].iloc[i], df['atr'].iloc[i]
            
            # Calculate NPRV
            nprv = calculate_nprv(current_price, high52, low52, range52, current_atr)
            
            # Evaluate market conditions
            score, _, _ = evaluate_market_conditions(df, i, num_data_points, adjacent_segments)

            if position == 0:
                if score > 0.00022 and nprv is not None and nprv > 0.5:  # Adjust these thresholds as needed
                    current_chunk = equity_manager.get_available_chunk()
                    if current_chunk:
                        position = int(current_chunk // current_price)
                        buy_price = current_price
                        profit_tiers, fibonacci_levels = calculate_tiers_and_levels(buy_price, current_atr)
                        total_profit -= max(MIN_COMMISSION, min(position * PER_SHARE_COMMISSION, position * current_price * MAX_COMMISSION_PERCENT))
                        total_trades += 1
                    
            elif position > 0:
                # Check for stop loss
                if current_price <= buy_price * (1 - stop_loss_percent):
                    profit = (current_price - buy_price) * position
                    total_profit += profit - max(MIN_COMMISSION, min(position * PER_SHARE_COMMISSION, position * current_price * MAX_COMMISSION_PERCENT))
                    equity_manager.release_chunk(position * current_price)
                    position = 0
                else:
                    # Check for profit taking
                    for tier, (mid_point, fib_level) in zip(profit_tiers[1:], fibonacci_levels):
                        if current_price >= mid_point:
                            profit = (current_price - buy_price) * position
                            total_profit += profit - max(MIN_COMMISSION, min(position * PER_SHARE_COMMISSION, position * current_price * MAX_COMMISSION_PERCENT))
                            equity_manager.release_chunk(position * current_price)
                            position = 0
                            break
                    
                    # Additional exit condition based on market conditions
                    if score < -0.00018:  # Adjust this threshold as needed
                        profit = (current_price - buy_price) * position
                        total_profit += profit - max(MIN_COMMISSION, min(position * PER_SHARE_COMMISSION, position * current_price * MAX_COMMISSION_PERCENT))
                        equity_manager.release_chunk(position * current_price)
                        position = 0

            # End of day processing
            if i < len(df) - 1 and is_end_of_day(df.index[i], df.index[i+1]):
                if position > 0:
                    profit = (current_price - buy_price) * position
                    total_profit += profit - max(MIN_COMMISSION, min(position * PER_SHARE_COMMISSION, position * current_price * MAX_COMMISSION_PERCENT))
                    equity_manager.release_chunk(position * current_price)
                    position = 0

        # Close any remaining position at the end of the backtest period
        if position > 0:
            final_price = df['close'].iloc[-1]
            profit = (final_price - buy_price) * position
            total_profit += profit - max(MIN_COMMISSION, min(position * PER_SHARE_COMMISSION, position * final_price * MAX_COMMISSION_PERCENT))
            equity_manager.release_chunk(position * final_price)

        return total_trades, total_profit

    async def process_stocks(ticker_symbols, equity_manager, num_data_points, adjacent_segments, stop_loss_percent, start_date, end_date):
        """Process all stocks concurrently."""
        contracts = [Stock(symbol, 'SMART', 'USD') for symbol in ticker_symbols]
        historical_data = {
            contract.symbol: calculate_indicators(util.df(await ib.reqHistoricalDataAsync(
                contract, endDateTime=f'{end_date} 23:59:59', durationStr='3 D', barSizeSetting='1 min',
                whatToShow='MIDPOINT', useRTH=False)))
            for contract in contracts
        }

        tasks = []
        for contract, df in zip(contracts, historical_data.values()):
            tasks.append(process_stock(contract, equity_manager, num_data_points, adjacent_segments, stop_loss_percent, df))

        results = await asyncio.gather(*tasks)
        return {contract.symbol: result for contract, result in zip(contracts, results)}

    async def main(ticker_symbols, equity_to_invest, num_data_points, adjacent_segments, stop_loss_percent, start_date, end_date):
        """Main entry point."""
        try:
            await ib.connectAsync('127.0.0.1', 7497, clientId=10)
            equity_manager = EquityManager(equity_to_invest)

            results = await process_stocks(ticker_symbols, equity_manager, num_data_points, adjacent_segments, stop_loss_percent, start_date, end_date)

            result_window = tk.Toplevel()
            result_window.title("Backtest Results")
            tree = ttk.Treeview(result_window, columns=('Ticker', 'Trades', 'Profit'), show='headings')
            tree.heading('Ticker', text='Ticker')
            tree.heading('Trades', text='Trades')
            tree.heading('Profit', text='Profit')
            tree.pack(padx=10, pady=10, expand=True, fill='both')

            for ticker, (trades, profit) in results.items():
                tree.insert('', 'end', values=(ticker, trades, f"${profit:.2f}"))

        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            ib.disconnect()

    class BacktestGUI:
        def __init__(self, master):
            self.master = master
            master.title("Backtest Configuration")
            master.geometry("500x500")
            self.create_widgets()

        def create_widgets(self):
            # Ticker Symbols
            ttk.Label(self.master, text="Ticker Symbols (comma-separated):").pack(pady=5)
            self.ticker_entry = ttk.Entry(self.master, width=40)
            self.ticker_entry.insert(0, "MYSZ")
            self.ticker_entry.pack(pady=5)

            # Equity to Invest
            ttk.Label(self.master, text="Equity to Invest:").pack(pady=5)
            self.equity_entry = ttk.Entry(self.master)
            self.equity_entry.insert(0, "1000")
            self.equity_entry.pack(pady=5)

            # Number of Data Points
            ttk.Label(self.master, text="Number of Data Points:").pack(pady=5)
            self.data_points_entry = ttk.Entry(self.master)
            self.data_points_entry.insert(0, "80")
            self.data_points_entry.pack(pady=5)

            # Adjacent Segments
            ttk.Label(self.master, text="Adjacent Segments:").pack(pady=5)
            self.segments_entry = ttk.Entry(self.master)
            self.segments_entry.insert(0, "30")
            self.segments_entry.pack(pady=5)

            # Stop Loss Percent
            ttk.Label(self.master, text="Stop Loss %:").pack(pady=5)
            self.stop_loss_entry = ttk.Entry(self.master)
            self.stop_loss_entry.insert(0, "0.02")
            self.stop_loss_entry.pack(pady=5)

            # Start Date
            ttk.Label(self.master, text="Start Date (YYYY-MM-DD):").pack(pady=5)
            self.start_date_entry = ttk.Entry(self.master)
            self.start_date_entry.insert(0, "20241227")
            self.start_date_entry.pack(pady=5)

            # End Date
            ttk.Label(self.master, text="End Date (YYYY-MM-DD):").pack(pady=5)
            self.end_date_entry = ttk.Entry(self.master)
            self.end_date_entry.insert(0, "20241227")
            self.end_date_entry.pack(pady=5)

            # Run Button
            self.run_button = ttk.Button(self.master, text="Run Backtest", command=self.run_backtest)
            self.run_button.pack(pady=20)

        def get_inputs(self):
            return {
                'ticker_symbols': [s.strip() for s in self.ticker_entry.get().split(',')],
                'equity_to_invest': float(self.equity_entry.get()),
                'num_data_points': int(self.data_points_entry.get()),
                'adjacent_segments': int(self.segments_entry.get()),
                'stop_loss_percent': float(self.stop_loss_entry.get()),
                'start_date': self.start_date_entry.get(),
                'end_date': self.end_date_entry.get()
            }

        def run_backtest(self):
            try:
                inputs = self.get_inputs()
                asyncio.run(main(**inputs))
            except ValueError as e:
                messagebox.showerror("Input Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")

    if __name__ == "__main__":
        root = tk.Tk()
        app = BacktestGUI(root)
        root.mainloop()
