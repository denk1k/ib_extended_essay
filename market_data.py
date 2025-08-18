import asyncio
import os
import aiohttp
import pandas as pd
import zipfile
import io
from datetime import datetime, timedelta
import json

from fetch_coins_and_filter import return_ticker_list
ASSETS_TO_QUERY = return_ticker_list()
def find_first_valid_date(df):
    if df.empty:
        return None

    df.index = pd.to_datetime(df.index)
    
    # date range from the first to the last day in the data
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    
    # dates in the full range that NOT in the df index
    missing_dates = full_range.difference(df.index)
    
    if missing_dates.empty:
        return df.index.min()

    # start of each gap
    s = missing_dates.to_series()
    gaps = s.diff().dt.days.ne(1).cumsum()
    
    # size
    gap_sizes = s.groupby(gaps).size()
    
    # first gap that is too long
    first_long_gap = gap_sizes[gap_sizes > 2].first_valid_index()

    if first_long_gap is None:
        return df.index.min() # valid

    else:
        print("First long gap found at day:", missing_dates[first_long_gap])

    # first valid date is after the first long gap ends
    last_day_of_gap = s[gaps == first_long_gap].max()
    first_valid_date_after_gap = df.index[df.index > last_day_of_gap].min()

    if pd.notna(first_valid_date_after_gap):
        return first_valid_date_after_gap
    else:
        # no data after first gap
        return None

async def download_and_process_data(session, symbol, day):
    # URL of the daily metrics data file
    url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"

    try:
        # asynchronously get the data with a timeout using an aiohttp session
        async with session.get(url, timeout=30.0) as response:
            response.raise_for_status()
            content = await response.read() # read as bytes

        # process ZIP in memory
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            # first file in the zip is the correct CSV
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)
                if df.empty:
                    print(f"Data for {symbol} on {day} is empty. Skipping.")
                    return None
                agg_rules = {
                    'sum_open_interest': lambda x: x.iloc[-1],
                    'sum_open_interest_value': lambda x: x.iloc[-1],
                    'count_toptrader_long_short_ratio': 'mean',
                    'sum_toptrader_long_short_ratio': 'mean',
                    'count_long_short_ratio': 'mean',
                    'sum_taker_long_short_vol_ratio': 'mean'
                }

                agg_columns = {col: agg_rules[col] for col in agg_rules if col in df.columns}

                if not agg_columns:
                    print(f"No aggregatable columns found for {symbol} on {day}. Skipping.")
                    return None

                aggregated_data = df[list(agg_columns.keys())].agg(agg_columns).to_dict()
                aggregated_data['date'] = day

                return aggregated_data

    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            # expected for dates with no data, so no error
            pass
        else:
            print(f"Error for {symbol} on {day}: {e.status} - {e.message}")
        return None
    except Exception as e:
        print(f"Error processing {symbol} on {day}, {e}")
        return None

async def main():
    ee = ASSETS_TO_QUERY
    symbols = [i + "USDT" for i in ee]
    renames = {"LUNCUSDT": "1000LUNCUSDT", "BEAMUSDT": "BEAMXUSDT", "XECUSDT":"1000XECUSDT", "LUNAUSDT": "LUNA2USDT"}
    start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')
    end_date = datetime.now()
    output_directory = 'market_data'
    stats_dir = 'data_fetching_stats'

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    market_start_dates = {}

    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            print(f"Starting processing for {symbol}")
            tasks = []
            current_date = start_date
            
            symbol_to_fetch = renames.get(symbol, symbol)

            while current_date <= end_date:
                day_str = current_date.strftime('%Y-%m-%d')
                task = download_and_process_data(session, symbol_to_fetch, day_str)
                tasks.append(task)
                current_date += timedelta(days=1)

            results = await asyncio.gather(*tasks)
            daily_records = [r for r in results if r is not None]

            if not daily_records:
                print(f"No data successfully processed for {symbol}. No file will be created.")
                continue

            final_df = pd.DataFrame(daily_records)
            final_df = final_df.set_index('date').sort_index()
            
            # first valid start date based on data continuity
            valid_start_date = find_first_valid_date(final_df)
            
            if valid_start_date:
                asset_symbol = symbol.replace('USDT', '')
                market_start_dates[asset_symbol] = valid_start_date.strftime('%Y-%m-%d')
                print(f"Found valid start date for {asset_symbol}: {market_start_dates[asset_symbol]}")
                
                # ffill the df from the valid start date
                final_df.index = pd.to_datetime(final_df.index)
                final_df_ffilled = final_df.reindex(pd.date_range(start=final_df.index.min(), end=final_df.index.max(), freq='D')).ffill(limit=2)
                final_df_cleaned = final_df_ffilled.loc[valid_start_date:].dropna(how='all')

                output_file = os.path.join(output_directory, f"{symbol}_aggregated_daily_metrics.csv")
                final_df_cleaned.to_csv(output_file)
                print(f"saved {len(final_df_cleaned)} records for {symbol} to {output_file}")
            else:
                print(f"No valid continuous data block found for {symbol}. No file will be created.")

    stats_filepath = os.path.join(stats_dir, 'market_data_start_dates.json')
    with open(stats_filepath, 'w') as f:
        json.dump(market_start_dates, f, indent=4)
    print(f"\nSaved market data start dates to {stats_filepath}")

if __name__ == '__main__':
    asyncio.run(main())
