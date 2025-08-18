import os
import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
from datetime import datetime, timedelta, timezone
import pyarrow.feather as feather
import json

async def download_past_data(pairs, exchange_name, timeframe_minutes, day_count=400, renames={}):
    if not pairs:
        print("No pairs provided. Exiting.")
        return

    userdir = './price_data'

    try:
        exchange_class = getattr(ccxt_async, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        print(f"CCXT exchange: {exchange_name}")
    except AttributeError:
        print(f"Error: '{exchange_name}' not found in ccxt.")
        return
    except Exception as e:
        print(f"Error on init {exchange_name}: {e}")
        return

    await exchange.load_markets()

    datetime_now = datetime.now(timezone.utc)
    end_ts = exchange.parse8601(datetime_now.isoformat())
    datetime_prev = datetime_now - timedelta(days=day_count)
    overall_start_ts = exchange.parse8601(datetime_prev.isoformat())

    if timeframe_minutes >= 1440 and timeframe_minutes % 1440 == 0:
        timeframe_str = f'{timeframe_minutes // 1440}d'
    elif timeframe_minutes >= 60 and timeframe_minutes % 60 == 0:
        timeframe_str = f'{timeframe_minutes // 60}h'
    else:
        timeframe_str = f'{timeframe_minutes}m'

    if timeframe_str not in exchange.timeframes:
        print(f"Error: Timeframe '{timeframe_str}' not supported by {exchange_name}.")
        await exchange.close()
        return

    limit = 1000
    timeframe_ms = exchange.parse_timeframe(timeframe_str) * 1000

    async def fetch_pair_data(pair):
        if pair not in exchange.markets:
            print(f"'{pair}' not found in {exchange_name} markets. Skipping.")
            return


        pair_official = pair
        if pair.split("/")[0] in renames:
            pair_official = pair.replace(pair.split("/")[0], renames.get(pair.split("/")[0], pair.split("/")[0]))

        safe_pair = pair_official.replace('/', '_').replace(':', '_')

        filename = os.path.join(userdir, f'{safe_pair}-{timeframe_str}-futures.feather')

        all_ohlcv = []
        current_since = overall_start_ts

        while current_since < end_ts:
            ohlcv = await exchange.fetch_ohlcv(pair, timeframe_str, since=current_since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1  # Go to next timestamp

        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)
            df.set_index('date', inplace=True)
            df = df[~df.index.duplicated(keep='first')].sort_index()

            # resample to continuous index
            freq = f'{timeframe_minutes // 1440}D' if timeframe_minutes % 1440 == 0 else f'{timeframe_minutes}T'
            full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz='UTC')
            df = df.reindex(full_index)

            # forward-fill small gaps (<=2 days)
            df['is_nan'] = df['close'].isna()  # Use 'close' as proxy for NaN rows
            df['nan_group'] = (df['is_nan'] != df['is_nan'].shift()).cumsum()
            nan_groups = df[df['is_nan']].groupby('nan_group').size()

            # count large gaps that won't be filled
            large_gaps = nan_groups[nan_groups > 2].index
            num_failed_fills = len(large_gaps)
            print(f"{pair}: {num_failed_fills} large gaps (>2 days) not forward-filled.")

            small_gaps = nan_groups[nan_groups <= 2].index  # 2 days max
            for group in small_gaps:
                mask = df['nan_group'] == group
                df.loc[mask] = df.loc[mask].ffill()

            df = df.drop(columns=['is_nan', 'nan_group'])


            # start from the last non-null after a null
            last_nan_idx = df[df['close'].isna()].index.max()
            if pd.notna(last_nan_idx):
                truncate_start = df.index[df.index > last_nan_idx][0] if len(df.index > last_nan_idx) > 0 else df.index[-1]
                df = df.loc[truncate_start:]
            df = df.dropna() # final drop of remaining nans

            if not df.empty:
                os.makedirs(userdir, exist_ok=True)
                feather.write_feather(df, filename)
                print(f"Saved {len(df)} candles (after gap handling and truncation) for {pair} to {filename}")
            else:
                print(f"No valid data after processing for {pair}. Skipping save.")
        else:
            print(f"No data fetched for {pair}. Skipping save.")

    tasks = [fetch_pair_data(pair) for pair in pairs]
    await asyncio.gather(*tasks)

    # price start dates JSON
    print("\n--- Generating Price Start Dates Summary ---")
    start_dates = {}
    stats_dir = 'data_fetching_stats'
    os.makedirs(stats_dir, exist_ok=True)

    for pair in pairs:

        if pair.split("/")[0] in renames:
            pair = pair.replace(pair.split("/")[0], renames.get(pair.split("/")[0], pair.split("/")[0]))

        safe_pair = pair.replace('/', '_').replace(':', '_')
        filename = os.path.join(userdir, f'{safe_pair}-{timeframe_str}-futures.feather')
        if os.path.exists(filename):
            try:
                df = feather.read_feather(filename)
                if not df.empty:
                    asset_symbol = pair.split('/')[0]
                    start_date = df.index.min().strftime('%Y-%m-%d')
                    start_dates[asset_symbol] = start_date
            except Exception as e:
                print(f"Could not process file {filename} for start date: {e}")

    stats_filepath = os.path.join(stats_dir, 'price_start_dates.json')
    with open(stats_filepath, 'w') as f:
        json.dump(start_dates, f, indent=4)
    print(f"Saved price start dates to {stats_filepath}")

    print('\nDownload process complete.')
    await exchange.close()

if __name__ == "__main__":
    renames = {"1000LUNC": "LUNC", "BEAMX": "BEAM", "1000XEC": "XEC",  "LUNA2":"LUNA"} # renames have to be manually determined unfortunately
    from fetch_coins_and_filter import return_ticker_list
    ASSETS_TO_QUERY = return_ticker_list()
    for final, initial in renames.items():
        if initial in ASSETS_TO_QUERY:
            ASSETS_TO_QUERY.remove(initial) # ensure Binance naming conventions
            ASSETS_TO_QUERY.append(final)
    asyncio.run(download_past_data(pairs=[i + "/USDT:USDT" for i in ASSETS_TO_QUERY],
                                       exchange_name='binance',
                                       timeframe_minutes=1440,
                                       day_count=5000, renames=renames)) # passing since they have to be renamed back
