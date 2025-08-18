#!/usr/bin/env bash
set -euo pipefail

ask_yes_no() {
  local prompt="$1"
  local default="${2:-Y}"
  local choice

  while true; do
    if [[ "$default" == "Y" ]]; then
      read -rp "$prompt [Y/n]: " choice || true
      choice="${choice:-Y}"
    else
      read -rp "$prompt [y/N]: " choice || true
      choice="${choice:-N}"
    fi
    case "$(echo "$choice" | tr '[:lower:]' '[:upper:]')" in
      Y|YES) return 0 ;;
      N|NO)  return 1 ;;
      *) echo "Please answer Y or N." ;;
    esac
  done
}

ask_option() {
  local prompt="$1"
  local opt1="$2"
  local opt2="$3"
  local choice
  while true; do
    read -rp "$prompt ($opt1/$opt2): " choice || true
    choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')
    if [[ "$choice" == "$opt1" || "$choice" == "$opt2" ]]; then
      echo "$choice"
      return 0
    fi
    echo "Please type exactly: $opt1 or $opt2"
  done
}

echo "1: Creating venv environment and getting dependencies"
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source ./.venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo ""
echo "You can:"
echo "-Reuse the frozen feature_data used in the essay (faster, skip to experiment), OR"
echo "-Fetch data and build features from scratch (slower, +~15 minutes)"

if ask_yes_no "Reuse feature_data directory?" "Y"; then
  USE_FROZEN_FEATURES=1
else
  USE_FROZEN_FEATURES=0
fi

if [[ "$USE_FROZEN_FEATURES" -eq 0 ]]; then
  echo ""
  echo "2: Filtering assets by availability (CryptoCompare/top-500) ~30s"
  python3 fetch_coins_and_filter.py

  echo ""
  echo "3: Fetching market, social, and price data (in parallel) ~15min"
  python3 market_data.py & PID1=$!
  python3 social_data.py & PID2=$!
  python3 price_data.py  & PID3=$!
  wait $PID1 $PID2 $PID3

  echo ""
  echo "4: Building features ~5s"
  python3 create_features.py
fi

echo ""
echo "Select horizon mode for experiment.py"
HMODE=$(ask_option "Run on limited or all horizons?" "ltd" "all")
echo "Selected mode: $HMODE"

echo ""
echo "5: Running experiment.py (~2 hours depending on machine)"
python3 experiment.py "$HMODE"

echo "6: Running experiment_buy_hold.py (~3s)"
python3 experiment_buy_hold.py

echo ""
echo "Done. Results shall be printed to the terminal."
