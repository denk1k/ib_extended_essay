import pandas as pd
import os
import numpy as np
FEATURE_DATA_DIR = 'feature_data'
# COINS = ["IOTX", "AXS", "FET", "LDO", "BAND", "FIL", "LRC", "LUNC", "LUNA", "API3", "ONE", "GAS", "INJ", "ZRX", "RSR", "ACH", "LPT", "T", "1INCH", "GRT", "ICP", "KNC", "ONT", "JOE", "XVG", "SOL", "CTSI", "HOT", "TWT", "ETC", "CFX", "HBAR", "YFI", "CAKE", "ATOM", "VET", "CRV", "BNB", "MINA", "EGLD", "BNT", "WOO", "ZEC", "RVN", "STX", "AVAX", "LINK", "BEAM", "RUNE", "AAVE", "NEO", "WAXP", "ROSE", "CKB", "ALGO", "SKL", "TRX", "SAND", "XTZ", "CELR", "MASK", "DOT", "XEC", "NEAR"]
from find_start_date import analyze_start_dates
COINS = analyze_start_dates()
TRAIN_SIZE = 0.7
INITIAL_BALANCE = 10000
TRANSACTION_COST = 0.005

def analyze_profit(df, signal_col, initial_balance=10000, transaction_cost=0.005, risk_fraction=1.0):
    df = df.copy()
    df['signal'] = df[signal_col].clip(-1, 1)

    cash = initial_balance
    position_size = 0.0
    portfolio_values = []

    for i in range(len(df)):
        start_of_period_value = portfolio_values[-1] if i > 0 else initial_balance
        current_price = df['close'].iloc[i]
        current_signal = df['signal'].iloc[i]

        target_position_value = start_of_period_value * current_signal * risk_fraction
        current_position_value = position_size * current_price
        trade_value = target_position_value - current_position_value

        cash -= abs(trade_value) * transaction_cost
        cash -= trade_value
        position_size = target_position_value / current_price

        end_of_period_value = cash + position_size * current_price
        portfolio_values.append(end_of_period_value)

    df['portfolio_value'] = portfolio_values
    df['daily_return'] = df['portfolio_value'].pct_change().fillna(0)

    # Performance metrics
    final_value = df['portfolio_value'].iloc[-1]
    total_profit = final_value - initial_balance
    roi = (total_profit / initial_balance)

    mean_daily_return = df['daily_return'].mean()
    std_daily_return = df['daily_return'].std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(365) if std_daily_return != 0 else 0

    annualized_return = roi
    cumulative_max = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0

    return {
        'Final Balance': final_value,
        'Total Profit ($)': total_profit,
        'Return on Investment (%)': roi * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown (%)': abs(max_drawdown) * 100
    }

all_results = []
for coin in COINS:
    feature_filename = f"{coin.upper()}_features.feather"
    feature_path = os.path.join(FEATURE_DATA_DIR, feature_filename)
    if not os.path.exists(feature_path):
        print(f"Feature file for {coin} not found!")
        continue

    df = pd.read_feather(feature_path)
    if 'date' in df.columns:
        df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Only use test set (assume 70/30 split)
    split_index = int(len(df) * TRAIN_SIZE)
    test_df = df.iloc[split_index:]

    # Always fully invested (signal = 1)
    test_df = test_df.copy()
    test_df['hold_signal'] = 1.0

    metrics = analyze_profit(test_df, 'hold_signal',
                             initial_balance=INITIAL_BALANCE,
                             transaction_cost=TRANSACTION_COST)
    result = {'coin': coin}
    result.update(metrics)
    all_results.append(result)

results_df = pd.DataFrame(all_results)
print(results_df[["coin", "Return on Investment (%)", "Final Balance", "Sharpe Ratio", "Max Drawdown (%)"]])
print("\nSummary for buy and hold (across coins):")
for column in results_df.columns:
    if column != "coin":
        print(f"{column} mean: {results_df[column].mean():.4f}")
        print(f"{column} median: {results_df[column].median():.4f}")
# print("\nSummary for buy and hold (across coins):")
# print("  - Arithmetic Mean ROI: %.4f%%" % results_df["Return on Investment (%)"].mean())
# print("  - Median ROI: %.4f%%" % results_df["Return on Investment (%)"].median())
# print("  - Arithmetic Mean Sharpe: %.4f" % results_df["Sharpe Ratio"].mean())
# print("  - Arithmetic Mean Sharpe: %.4f" % results_df["Sharpe Ratio"].mean())
# print("  - Median Sharpe: %.4f" % results_df["Sharpe Ratio"].median())
