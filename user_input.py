import requests
import pandas as pd
import matplotlib.pyplot as plt

# Alpha Vantage API key
API_KEY = 'YOUR_API_KEY'
BASE_URL = 'https://www.alphavantage.co/query'

# Function to fetch historical data
def fetch_data(symbol, interval='60min', outputsize='full'):
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'outputsize': outputsize,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    df = pd.DataFrame.from_dict(data['Time Series (60min)'], orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# Function to calculate SMAs
def calculate_sma(data, short_window, long_window):
    data['short_sma'] = data['4. close'].rolling(window=short_window).mean()
    data['long_sma'] = data['4. close'].rolling(window=long_window).mean()
    return data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['4. close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    return data

# Trading strategy combining SMA and RSI
def combined_strategy(data):
    buy_signals = []
    sell_signals = []

    for i in range(1, len(data)):
        if (data['short_sma'].iloc[i] > data['long_sma'].iloc[i]) and (data['rsi'].iloc[i] < 50):
            buy_signals.append(data.index[i])
        elif (data['short_sma'].iloc[i] < data['long_sma'].iloc[i]) and (data['rsi'].iloc[i] > 50):
            sell_signals.append(data.index[i])

    return buy_signals, sell_signals

# Simulated portfolio
class Portfolio:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.shares = 0
        self.transaction_history = []
        self.portfolio_value = []
        self.buy_prices = []  # Store buy prices for calculating profits

    def buy(self, price, quantity, time):
        cost = price * quantity
        if self.balance >= cost:
            self.balance -= cost
            self.shares += quantity
            for _ in range(quantity):
                self.buy_prices.append(price)
            self.transaction_history.append(('buy', price, quantity, time))
            print(f"Bought {quantity} shares at {price} each at {time}")
        else:
            print("Insufficient balance to buy")

    def sell(self, price, quantity, time):
        if self.shares >= quantity:
            self.shares -= quantity
            self.balance += price * quantity
            avg_buy_price = sum(self.buy_prices[:quantity]) / quantity if self.buy_prices else 0
            del self.buy_prices[:quantity]
            self.transaction_history.append(('sell', price, quantity, time, avg_buy_price))
            print(f"Sold {quantity} shares at {price} each at {time}")
        else:
            print("Insufficient shares to sell")

    def get_value(self, current_price):
        return self.balance + self.shares * current_price

    def update_portfolio_value(self, current_price, time):
        self.portfolio_value.append((time, self.get_value(current_price)))

# Function to calculate performance metrics
def calculate_metrics(portfolio, initial_balance, data):
    final_value = portfolio.get_value(data['4. close'].iloc[-1])
    total_return = (final_value - initial_balance) / initial_balance * 100
    num_trades = len(portfolio.transaction_history)
    winning_trades = [t for t in portfolio.transaction_history if t[0] == 'sell' and t[1] > t[4]]
    losing_trades = [t for t in portfolio.transaction_history if t[0] == 'sell' and t[1] <= t[4]]
    win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
    avg_profit_per_trade = (final_value - initial_balance) / num_trades if num_trades > 0 else 0
    
    metrics = {
        'Final Portfolio Value': final_value,
        'Total Return (%)': total_return,
        'Number of Trades': num_trades,
        'Winning Trades': len(winning_trades),
        'Losing Trades': len(losing_trades),
        'Win Rate (%)': win_rate,
        'Average Profit per Trade': avg_profit_per_trade
    }
    
    return metrics

# Function to run backtest
def backtest(symbol, initial_balance, short_window, long_window, rsi_window):
    data = fetch_data(symbol, '60min', 'full')
    data = calculate_sma(data, short_window, long_window)
    data = calculate_rsi(data, rsi_window)

    buy_signals, sell_signals = combined_strategy(data)
    portfolio = Portfolio(initial_balance)

    for time in data.index:
        current_price = data['4. close'].loc[time]

        if time in buy_signals:
            quantity_to_buy = int(portfolio.balance // current_price)
            if quantity_to_buy > 0:
                portfolio.buy(current_price, quantity_to_buy, time)
        elif time in sell_signals:
            if portfolio.shares > 0:
                portfolio.sell(current_price, portfolio.shares, time)

        portfolio.update_portfolio_value(current_price, time)

    metrics = calculate_metrics(portfolio, initial_balance, data)
    return portfolio, data, metrics

# Function to plot results
def plot_results(portfolio, data, buy_signals, sell_signals):
    plt.figure(figsize=(14, 10))

    # Plot the closing prices and SMAs
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['4. close'], label='Close Price', color='black')
    plt.plot(data.index, data['short_sma'], label='Short SMA', color='blue')
    plt.plot(data.index, data['long_sma'], label='Long SMA', color='red')
    plt.scatter(buy_signals, data.loc[buy_signals]['4. close'], label='Buy Signal', marker='^', color='green')
    plt.scatter(sell_signals, data.loc[sell_signals]['4. close'], label='Sell Signal', marker='v', color='red')
    plt.legend()
    plt.title('Price and SMAs')

    # Plot the RSI
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['rsi'], label='RSI', color='purple')
    plt.axhline(y=70, color='red', linestyle='--')
    plt.axhline(y=30, color='green', linestyle='--')
    plt.legend()
    plt.title('RSI')

    # Plot the portfolio value over time
    plt.subplot(3, 1, 3)
    times, values = zip(*portfolio.portfolio_value)
    plt.plot(times, values, label='Portfolio Value', color='purple')
    plt.legend()
    plt.title('Portfolio Value Over Time')

    plt.tight_layout()
    plt.show()

def main():
    # Get user inputs
    symbol = input("Enter the stock symbol: ")
    initial_balance = float(input("Enter the initial balance (USD): "))
    short_window = int(input("Enter the short window for SMA: "))
    long_window = int(input("Enter the long window for SMA: "))
    rsi_window = int(input("Enter the window for RSI: "))

    # Run backtest
    portfolio, data, metrics = backtest(symbol, initial_balance, short_window, long_window, rsi_window)
    buy_signals, sell_signals = combined_strategy(data)

    # Plot results
    plot_results(portfolio, data, buy_signals, sell_signals)

    # Display performance metrics immediately after plotting
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()
