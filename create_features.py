from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import os
import json
from find_start_date import analyze_start_dates
# COINS = ["IOTX", "AXS", "FET", "LDO", "BAND", "FIL", "LRC", "LUNC", "LUNA", "API3", "ONE", "GAS", "INJ", "ZRX", "RSR", "ACH", "LPT", "T", "1INCH", "GRT", "ICP", "KNC", "ONT", "JOE", "XVG", "SOL", "CTSI", "HOT", "TWT", "ETC", "CFX", "HBAR", "YFI", "CAKE", "ATOM", "VET", "CRV", "BNB", "MINA", "EGLD", "BNT", "WOO", "ZEC", "RVN", "STX", "AVAX", "LINK", "BEAM", "RUNE", "AAVE", "NEO", "WAXP", "ROSE", "CKB", "ALGO", "SKL", "TRX", "SAND", "XTZ", "CELR", "MASK", "DOT", "XEC", "NEAR"]
COINS = analyze_start_dates()
PRICE_DATA_DIR = 'price_data'
SOCIAL_DATA_DIR = 'social_data'
MARKET_DATA_DIR = 'market_data'
OUTPUT_DIR = 'feature_data'
HEADROOM_DAYS = 200
INDICATOR_PERIODS = [10, 20, 50, 100]

def calculate_technical_indicators(df, periods):
    """
    Calculates TI on the price data.
    """
    print(f"Calculating indicators for {len(df)} rows...")

    for p in periods:
        # SMA
        df[f'sma_{p}'] = df['close'].rolling(window=p).mean()

        # EMA
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()

        # Rate of Change
        df[f'rocr_{p}'] = (df['close'] / df['close'].shift(p)) - 1

        # Bollinger Bands
        sma = df[f'sma_{p}']
        std = df['close'].rolling(window=p).std()
        df[f'bb_upper_{p}'] = sma + (std * 2)
        df[f'bb_lower_{p}'] = sma - (std * 2)

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ema_gain = gain.ewm(com=p - 1, adjust=False).mean()
        ema_loss = loss.ewm(com=p - 1, adjust=False).mean()
        rs = ema_gain / ema_loss
        df[f'rsi_{p}'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    print("Done with TIs.")
    return df

# Main Script
def create_feature_datasets():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # latest start date across all assets
    all_latest_starts = []
    all_earliest_ends = []
    for coin in COINS:
        try:
            price_pair_name = f"{coin.upper()}_USDT_USDT"
            price_filename = f"{price_pair_name}-1d-futures.feather"
            price_path = os.path.join(PRICE_DATA_DIR, price_filename)
            price_df = pd.read_feather(price_path)
            if 'date' in price_df.columns: price_df.set_index('date', inplace=True)
            price_df.index = pd.to_datetime(price_df.index, utc=True)

            social_filename = f"{coin.upper()}_social_trimmed.feather"
            social_path = os.path.join(SOCIAL_DATA_DIR, social_filename)
            social_df = pd.read_feather(social_path)
            date_column = 'date' if 'date' in social_df.columns else 'timestamp'
            if date_column: social_df.set_index(date_column, inplace=True)
            social_df.index = pd.to_datetime(social_df.index, utc=True)

            market_symbol = f"{coin.upper()}USDT"
            market_filename = f"{market_symbol}_aggregated_daily_metrics.csv"
            market_path = os.path.join(MARKET_DATA_DIR, market_filename)
            market_df = pd.read_csv(market_path, index_col=0) # , parse_dates=True
            # print(market_df)
            # market_df.set_index('date', inplace=True)
            # market_df.index = pd.to_datetime(market_df.index)
            market_df.index = pd.to_datetime(market_df.index, utc=True)

            coin_start_date = max(price_df.index.min(), social_df.index.min(), market_df.index.min())
            all_latest_starts.append(coin_start_date)
            coin_end_date = min(price_df.index.max(), social_df.index.max(), market_df.index.max())
            all_earliest_ends.append(coin_end_date)
            print(f"Latest start for {coin}: {coin_start_date.date()}, end of data: {coin_end_date.date()}")
        except Exception as e:
            print(f"Error occurred while pre-checking {coin}: {e}. Aborting.")
            return


    if not all_latest_starts:
        print("Could not determine any start dates. Aborting.")
        return

    universal_start_date = max(all_latest_starts)
    universal_end_date = min(all_earliest_ends)
    print(f"Universal date range set to: {universal_start_date.date()} to {universal_end_date.date()}")

    for coin in COINS:
        print(f"Processing {coin}")

        try:
            print("Loading data sources...")
            # Price
            price_pair_name = f"{coin.upper()}_USDT_USDT"
            price_filename = f"{price_pair_name}-1d-futures.feather"
            price_path = os.path.join(PRICE_DATA_DIR, price_filename)
            price_df = pd.read_feather(price_path)
            if 'date' in price_df.columns: # Handle if index was reset
                price_df.set_index('date', inplace=True)
            if not isinstance(price_df.index, pd.DatetimeIndex):
                 raise ValueError(f"Price data for {coin} does not have a DatetimeIndex.")
            if price_df.index.tz is None:
                price_df.index = price_df.index.tz_localize('UTC')
            else:
                price_df.index = price_df.index.tz_convert('UTC')
            print(f"Price data loaded. Range: {price_df.index.min().date()} to {price_df.index.max().date()}")

            # Social
            social_filename = f"{coin.upper()}_social_trimmed.feather"
            social_path = os.path.join(SOCIAL_DATA_DIR, social_filename)
            social_df = pd.read_feather(social_path)

            # Find & set date column
            date_column = None
            if 'date' in social_df.columns:
                date_column = 'date'
            elif 'timestamp' in social_df.columns:
                date_column = 'timestamp'

            if date_column:
                social_df.set_index(date_column, inplace=True)

            if not isinstance(social_df.index, pd.DatetimeIndex):
                social_df.index = pd.to_datetime(social_df.index, errors='coerce')

            if not isinstance(social_df.index, pd.DatetimeIndex):
                raise ValueError(f"Social data for {coin} has an index that could not be converted to DatetimeIndex.")

            if social_df.index.tz is None:
                social_df.index = social_df.index.tz_localize('UTC')
            else:
                social_df.index = social_df.index.tz_convert('UTC')
            print(f"Social data loaded. Range: {social_df.index.min().date()} to {social_df.index.max().date()}")

            # Market
            market_symbol = f"{coin.upper()}USDT"
            market_filename = f"{market_symbol}_aggregated_daily_metrics.csv"
            market_path = os.path.join(MARKET_DATA_DIR, market_filename)
            market_df = pd.read_csv(market_path, index_col=0)
            market_df.index = pd.to_datetime(market_df.index)

            # market_df.set_index('date', inplace=True)
            if market_df.index.tz is None:
                market_df.index = market_df.index.tz_localize('UTC')
            else:
                market_df.index = market_df.index.tz_convert('UTC')
            print(f"Market data loaded. Range: {market_df.index.min().date()} to {market_df.index.max().date()}")
            # Common date range
            latest_start_date = universal_start_date
            earliest_end_date = min(price_df.index.max(), social_df.index.max(), market_df.index.max())

            if latest_start_date > earliest_end_date:
                print(f"Warning: No overlapping data found for {coin}. The latest start date ({latest_start_date.date()}) is after the earliest end date ({earliest_end_date.date()}). Skipping.")
                continue

            print(f"Common date range: {latest_start_date.date()} to {earliest_end_date.date()}")

            # TIs with headroom
            indicator_start_date = latest_start_date - pd.Timedelta(days=HEADROOM_DAYS)
            price_for_indicators = price_df[price_df.index >= indicator_start_date].copy()

            price_with_indicators = calculate_technical_indicators(price_for_indicators, INDICATOR_PERIODS)

            # Merge DataFrames
            print("Merging dataframes...")
            final_df = pd.concat([price_with_indicators, social_df, market_df], axis=1, join='inner')
            cols_to_drop = [col for col in final_df.columns if col.endswith(('_UNIT', '_TYPE', '_ASSET_ID', '_ASSET_SYMBOL'))]
            if cols_to_drop:
                print(f"Removing {len(cols_to_drop)} columns with UNIT/TYPE/ASSET_ID/ASSET_SYMBOL suffixes.")
                final_df.drop(columns=cols_to_drop, inplace=True)

            # drop any rows with NaN values
            initial_rows = len(final_df)
            final_df = final_df.ffill()
            final_df.dropna(inplace=True)
            if len(final_df) < initial_rows:
                print(f"Dropped {initial_rows - len(final_df)} rows due to NaN values after merge.")

            # Trim to a common date range
            final_df = final_df[(final_df.index >= universal_start_date) & (final_df.index <= universal_end_date)]
            if final_df.empty:
                print(f"Warning: The final merged dataframe for {coin} is empty after filtering for the common date range.")
                continue

            # Save
            output_filename = f"{coin.upper()}_features.feather"
            # output_filename1 = f"{coin.upper()}_features.csv"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            # output_path1 = os.path.join(OUTPUT_DIR, output_filename1)
            final_df.to_feather(output_path)
            # final_df.to_csv(output_path1)

            print(f"Created feature dataset for {coin} with {len(final_df)} rows.")
            print(f"Saved to {output_path}")
            print("Sample of final data:")
            print(final_df.head())
            print("\n")
        except Exception as e:
            print(f"Error occurred while processing {coin}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    create_feature_datasets()
