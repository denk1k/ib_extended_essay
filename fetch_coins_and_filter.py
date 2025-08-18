import requests
import json
from typing import Dict, List

class CryptoCompareChecker:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://data-api.cryptocompare.com"
        self.headers = {}

    def get_top_by_market_cap(self, limit: int = 200):
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        while self.api_key is None:
            self.api_key = input("Put CoinMarketCap API key here:")
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.api_key,
        }

        params = {
            'start': '1',
            'limit': str(limit),
            'convert': 'USD'
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            print(f"{e} while fetching top cryptos")
            return []

    def get_cc_social_data(self):
        url = f"{self.base_url}/asset/v1/summary/list"
        params = {
            'filters': 'HAS_DISCORD_SERVERS,HAS_SUBREDDITS,HAS_CODE_REPOSITORIES,HAS_TELEGRAM_GROUPS,HAS_TWITTER'
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('Data', {})
        except Exception as e:
            print(f"Error while fetching cc social data: {e}")
            return {}

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.upper().strip()

    def check_coverage(self, top_cryptos: List[Dict], social_data: Dict) -> Dict:
        social_symbols = set()
        for item in social_data.values():
            if isinstance(item, list):
                for crypto_entry in item:
                    if isinstance(crypto_entry, dict):
                        symbol = crypto_entry.get('SYMBOL', '')
                        if symbol:
                            social_symbols.add(self.normalize_symbol(symbol))
            elif isinstance(item, dict):
                symbol = item.get('SYMBOL', '')
                if symbol:
                    social_symbols.add(self.normalize_symbol(symbol))
        with_social = []
        without_social = []
        for crypto in top_cryptos:
            symbol = self.normalize_symbol(crypto.get('symbol', ''))
            name = crypto.get('name', '')
            market_cap = crypto.get('quote', {}).get('USD', {}).get('market_cap', 0)
            rank = crypto.get('cmc_rank', 0)
            crypto_info = {
                'symbol': symbol,
                'name': name,
                'market_cap': market_cap,
                'rank': rank
            }
            if symbol in social_symbols:
                with_social.append(crypto_info)
            else:
                without_social.append(crypto_info)

        return {
            'with_social': with_social,
            'without_social': without_social,
            'total_checked': len(top_cryptos),
            'coverage_percentage': (len(with_social) / len(top_cryptos)) * 100 if top_cryptos else 0
        }


    def print_results(self, results):
        print(f"\nTotal analyzed: {results['total_checked']}")
        print(f"Coverage: {results['coverage_percentage']:.1f}%")
        print(f"With social data: {len(results['with_social'])}")
        print(f"Without social data: {len(results['without_social'])}")
        print("TOP CRYPTOCURRENCIES WITH SOCIAL DATA")
        for crypto in results['with_social']:
            print(f"#{crypto['rank']:3d} {crypto['symbol']:8s} - {crypto['name']:30s} ${crypto['market_cap']:,.0f}")
        print("TOP CRYPTOCURRENCIES WITHOUT SOCIAL DATA:")

        for crypto in results['without_social']:
            print(f"#{crypto['rank']:3d} {crypto['symbol']:8s} - {crypto['name']:30s} ${crypto['market_cap']:,.0f}")

        print("available",json.dumps([crypto['symbol'] for crypto in results['with_social']]))

    def save_results_to_file(self, results: Dict, filename: str = "data_availability.json") -> None:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to {filename}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    checker = CryptoCompareChecker(api_key=None)
    print("Fetching top 500 by market cap...")
    top_cryptos = checker.get_top_by_market_cap(500)
    if not top_cryptos:
        print("Failed to fetch.")
        return
    print(f"Fetched {len(top_cryptos)} cryptos")
    social_data = checker.get_cc_social_data()
    if not social_data:
        print("Failed to fetch.")
        return
    print(f"Fetched social data for {len(social_data)} cryptocurrencies")
    results = checker.check_coverage(top_cryptos, social_data)
    checker.print_results(results)
    checker.save_results_to_file(results)

def return_ticker_list(file_name: str = "data_availability.json"):
    """Helper function for the price, market and social data fetching scripts"""
    with open("data_availability.json", 'r') as f:
        ticker_info = json.load(f)
        ticker_list = [i["symbol"] for i in ticker_info['with_social']]
        print(f"Got {len(ticker_list)} tickers to fetch")
        return ticker_list # no error handling here, since it is better to crash the run if there are no tickers available

if __name__ == "__main__":
    main()
