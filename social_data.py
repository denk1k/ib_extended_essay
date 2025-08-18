import requests
import pandas as pd
import time
import os
import json
BASE_URL = "https://data-api.coindesk.com"
BASE_PATH = "/asset/v1/historical"

ENDPOINTS = {
    "twitter": "/twitter/days",
    "reddit": "/reddit/days",
    "discord": "/discord/days",
    "code_repository": "/code-repository/days",
    "telegram": "/telegram/days"
}
from fetch_coins_and_filter import return_ticker_list
ASSETS_TO_QUERY = return_ticker_list()

ACTIVITY_METRICS = {
    "twitter": "TOTAL_FOLLOWERS",
    "reddit": "TOTAL_SUBSCRIBERS",
    "discord": "TOTAL_MEMBERS",
    "telegram": "TOTAL_MEMBERS",
    "code_repository": "TOTAL_STARS"
}

HEADERS = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Origin': 'https://developers.coindesk.com',
    'Referer': 'https://developers.coindesk.com/',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
}

PARAMS = {
    "groups": "ID,GENERAL,ACTIVITY",
    "limit": 2000,
    "aggregate": 1,
    "fill": "true",
    "response_format": "JSON",
    "asset_lookup_priority": "SYMBOL"
}
SAVE_DIRECTORY = "social_data"


def fetch_coindesk_data(asset_symbol, endpoint_name):
    full_url = f"{BASE_URL}{BASE_PATH}{ENDPOINTS[endpoint_name]}"
    query_params = {**PARAMS, "asset": asset_symbol}

    print(f"Getting data for {asset_symbol} from {endpoint_name}...")
    try:
        response = requests.get(full_url, params=query_params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        if 'Data' in data and data['Data']:
            return pd.DataFrame(data['Data'])
        else:
            print(f"Warning: No data returned for {asset_symbol} from {endpoint_name}.")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch data for {asset_symbol} from {endpoint_name}. Reason: {e}")
        return pd.DataFrame()

def get_common_start_date(df, endpoint_prefixes):
    start_dates = []
    max_allowed_gap = pd.Timedelta(days=2)


    for endpoint in endpoint_prefixes:
        long_gap_detected = False
        metric_col = f"{endpoint}_{ACTIVITY_METRICS[endpoint]}"
        if metric_col not in df.columns:
            print(f"'{metric_col}' not found for endpoint. Asset will be excluded.")
            return None

        # Find first index >0 (non-zero & non-null)
        valid_mask = (df[metric_col] > 0) & df[metric_col].notnull()
        non_zero_idx = df.index[valid_mask]

        if non_zero_idx.empty:
            print(f"No activity (all zeros or nulls) found for {metric_col}. Asset will be excluded.")
            return None

        first_active_date = non_zero_idx.min()

        # Check rest of the series after first_active_date for gaps
        series = df.loc[first_active_date:, metric_col]
        series = series[~(series.isnull() | (series == 0))]  # drop further zero/nulls if any
        series_full = series.resample('D').asfreq()

        # Find missing values (=NaN)
        na_locs = series_full[series_full.isnull()].index

        # If NA locations there check gaps
        gap_starts = []
        last_non_na = None
        for dt in series_full.index:
            if pd.isna(series_full[dt]):
                if last_non_na is not None:
                    gap_start = last_non_na
                    # find out how long until the next non-NA
                    next_vals = series_full[dt:].dropna()
                    if not next_vals.empty:
                        next_non_na = next_vals.index[0]
                        gap = next_non_na - gap_start
                        if gap > max_allowed_gap:
                            print(f"Long data gap of {gap.days} days detected for {metric_col}, starting at {gap_start.date()}. Data will be truncated to begin after this gap.")
                            long_gap_detected = True
                            # new start as the next non-null after long gap
                            first_active_date = next_non_na
                    break  # Truncate at first long gap
            else:
                last_non_na = dt

        start_dates.append(first_active_date)
        print(f"Start for {metric_col}: {first_active_date.date()} (long gap detected: {long_gap_detected})")

    if len(start_dates) != len(endpoint_prefixes):
        print("Mismatch between found start dates and expected streams. Asset will be excluded.")
        return None

    common_start_date = max(start_dates)
    print(f"Common start date for analysis (after gap handling): {common_start_date.date()}")
    return common_start_date

def main():
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    stats_dir = 'data_fetching_stats'
    os.makedirs(stats_dir, exist_ok=True)
    
    social_start_dates = {}
    excluded_assets = []

    for asset in ASSETS_TO_QUERY:
        print(f"\nProcessing Asset: {asset}")
        asset_dfs = {}
        
        all_endpoints_fetched = True
        for endpoint_name in ENDPOINTS.keys():
            df = fetch_coindesk_data(asset, endpoint_name)
            if df.empty:
                print(f"Excluding {asset}: No data for endpoint '{endpoint_name}'.")
                all_endpoints_fetched = False
                break
            
            df = df.rename(columns={"TIMESTAMP": "timestamp"}).set_index("timestamp")
            df = df.add_prefix(f"{endpoint_name}_")
            asset_dfs[endpoint_name] = df
            time.sleep(1)

        if not all_endpoints_fetched:
            excluded_assets.append(asset)
            continue

        endpoint_keys = list(asset_dfs.keys())
        merged_df = asset_dfs[endpoint_keys[0]]
        for key in endpoint_keys[1:]:
            merged_df = merged_df.join(asset_dfs[key], how='outer')

        merged_df.index = pd.to_datetime(merged_df.index, unit='s')
        merged_df = merged_df.sort_index()
        merged_df = merged_df.ffill(limit=2)

        common_start_date = get_common_start_date(merged_df, list(asset_dfs.keys()))

        if common_start_date is None:
            excluded_assets.append(asset)
            continue

        trimmed_df = merged_df.loc[common_start_date:]
        
        output_path = os.path.join(SAVE_DIRECTORY, f"{asset}_social_trimmed.feather")
        trimmed_df.reset_index().rename(columns={'index': 'date'}).to_feather(output_path)
        print(f"saved trimmed social data for {asset} to {output_path}\n")
        
        social_start_dates[asset] = common_start_date.strftime('%Y-%m-%d')

    stats_filepath = os.path.join(stats_dir, 'social_start_dates.json')
    with open(stats_filepath, 'w') as f:
        json.dump({'start_dates': social_start_dates, 'excluded_assets': excluded_assets}, f, indent=4)
    print(f"\nSaved start dates and exclusions to {stats_filepath}")
    print("EXCLUDED")
    print(", ".join(excluded_assets))

if __name__ == "__main__":
    main()