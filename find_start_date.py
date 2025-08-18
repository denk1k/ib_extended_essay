import json
import os
from datetime import datetime, timedelta

def analyze_start_dates():
    stats_dir = 'data_fetching_stats'
    price_file = os.path.join(stats_dir, 'price_start_dates.json')
    social_file = os.path.join(stats_dir, 'social_start_dates.json')
    market_file = os.path.join(stats_dir, 'market_data_start_dates.json')
    try:
        with open(price_file, 'r') as f:
            price_data = json.load(f)
        print(f"Successfully loaded price start dates for {len(price_data)} assets.")
    except FileNotFoundError:
        print(f"Price data file not found at {price_file}")
        return
    except json.JSONDecodeError:
        print(f"Could not decode JSON from {price_file}")
        return

    try:
        with open(social_file, 'r') as f:
            social_data = json.load(f)
        social_start_dates = social_data.get('start_dates', {})
        social_excluded = social_data.get('excluded_assets', [])
        print(f"Successfully loaded social media start dates for {len(social_start_dates)} assets.")
        if social_excluded:
            print(f"Social media data streams were unavailable for: {', '.join(social_excluded)}")
    except FileNotFoundError:
        print(f"Error: Social data file not found at {social_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {social_file}")
        return

    try:
        with open(market_file, 'r') as f:
            market_data = json.load(f)
        print(f"Successfully loaded market data start dates for {len(market_data)} assets.")
    except FileNotFoundError:
        print(f"Market data file not found at {market_file}")
        return
    except json.JSONDecodeError:
        print(f"Could not decode JSON from {market_file}")
        return

    combined_dates = {}
    omitted_assets = []

    all_price_assets = set(price_data.keys())
    all_social_assets = set(social_start_dates.keys())
    all_market_assets = set(market_data.keys())
    omitted_binance = all_social_assets - all_price_assets
    if omitted_binance:
        print(f"\nAssets with social data but omitted from Binance price data: {', '.join(sorted(list(omitted_binance)))} ")

    omitted_market = all_price_assets.intersection(all_social_assets) - all_market_assets
    if omitted_market:
        print(f"Assets with price/social data but omitted cuz no market data: {', '.join(sorted(list(omitted_market)))} ")

    common_assets = all_price_assets.intersection(all_social_assets).intersection(all_market_assets)
    print(f"\nFound {len(common_assets)} common assets.")

    for asset in common_assets:
        price_date_str = price_data[asset]
        social_date_str = social_start_dates[asset]
        market_date_str = market_data[asset]
        
        price_date = datetime.strptime(price_date_str, '%Y-%m-%d')
        social_date = datetime.strptime(social_date_str, '%Y-%m-%d')
        market_date = datetime.strptime(market_date_str, '%Y-%m-%d')
        effective_start_date = max(price_date, social_date, market_date)
        combined_dates[asset] = effective_start_date

    if not combined_dates:
        print("No common assets with valid start dates found.")
        return
    sorted_dates = sorted(combined_dates.values())
    total_assets = len(combined_dates)
    target_availability_pct = 0.8
    
    optimal_date = None

    for i, current_date in enumerate(sorted_dates):
        available_assets_count = sum(1 for asset_date in sorted_dates if asset_date <= current_date)
        availability_pct = available_assets_count / total_assets

        if availability_pct >= target_availability_pct:
            if i + 1 < len(sorted_dates):
                next_asset_date = sorted_dates[i+1]
                if next_asset_date >= current_date + timedelta(days=30):
                    optimal_date = current_date
                    break # date found
            else:
                optimal_date = current_date
                break

    print("\nOptimal Start Date Analysis")
    if optimal_date:
        print(f"Optimal Start Date Found: {optimal_date.strftime('%Y-%m-%d')}")
        available_on_date = [asset for asset, date in combined_dates.items() if date <= optimal_date]
        unavailable_on_date = [asset for asset, date in combined_dates.items() if date > optimal_date]
        print(f"{len(available_on_date)}/{total_assets} ({len(available_on_date)/total_assets:.1%}) of common assets are available on this date.")
        print(f"Assets available: {', '.join(sorted(available_on_date))}")
        print(json.dumps(available_on_date))
        if unavailable_on_date:
            print(f"Assets not available until after the date: {', '.join(sorted(unavailable_on_date))}")
        return available_on_date
    else:
        print("Couldn't find optimal start date that meets all criteria.")
        print("Potentially, no date had 90% availability, or the date gap condition was never met.")


if __name__ == "__main__":
    # This script is not supposed to be run directly, that can be done though
    analyze_start_dates()
