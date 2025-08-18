import os
import subprocess
import venv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def ask(prompt):
    while True:
        choice = (input(f"{prompt} [Y/n]: ").strip() or "NONE").upper()
        if choice in ("Y", "YES"):
            return True
        if choice in ("N", "NO"):
            return False
        print("Please answer Y or N")

def asktext(prompt, opt1, opt2):
    opt1 = opt1.lower()
    opt2 = opt2.lower()
    while True:
        choice = input(f"{prompt} ({opt1}/{opt2}): ").strip().lower()
        if choice in (opt1, opt2):
            return choice
        print(f"Please type {opt1} or {opt2}")

def run(cmd, env=None):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def ensure_venv(venv_dir):
    if not venv_dir.exists():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir.as_posix())

def venv_python(venv_dir):
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"

def main():
    print("1: Creating venv and getting deps")
    root = Path(__file__).resolve().parent
    venv_dir = root / ".venv"
    ensure_venv(venv_dir)
    py = venv_python(venv_dir)

    run([py.as_posix(), "-m", "pip", "install", "--upgrade", "pip"])
    run([py.as_posix(), "-m", "pip", "install", "-r", "requirements.txt"])

    print("")
    print("You can:")
    print("-Reuse the frozen feature_data used in the essay (faster, skip to experiment)")
    print("-Fetch data and build features from scratch (slower, + ~30 minutes)")
    use_frozen = ask("Reuse feature_data?")

    if not use_frozen:
        print("")
        print("2: Filtering assets by availability (CryptoCompare/top-500) ~10s")
        run([py.as_posix(), "fetch_coins_and_filter.py"])

        print("")
        print("3: Fetching market, social, and price data (in parallel) ~30min")
        scripts = ["market_data.py", "social_data.py", "price_data.py"]
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(run, [py.as_posix(), s]) for s in scripts]
            for f in as_completed(futures):
                f.result()

        print("")
        print("4: Building features ~5s")
        run([py.as_posix(), "create_features.py"])

    print("")
    print("Select horizon mode for experiment.py")
    hmode = asktext("Run on limited or all horizons?", "ltd", "all")
    print(f"Selected mode: {hmode}")
    print("")
    print("5: Running experiment.py (~2 hours)")
    run([py.as_posix(), "experiment.py", hmode])

    print("6: Running experiment_buy_hold.py (~3s)")
    run([py.as_posix(), "experiment_buy_hold.py"])

    print("")
    print("Done. Results shall have been printed to the terminal.")

if __name__ == "__main__":
    main()
