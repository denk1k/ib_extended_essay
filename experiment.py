import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
from collections import defaultdict
from scipy.stats.mstats import gmean
from scipy.stats import wilcoxon
import sys
from io import StringIO

warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")
from find_start_date import analyze_start_dates
COINS = analyze_start_dates()
print(COINS)
FEATURE_DATA_DIR = 'feature_data'
TRAIN_SIZE = 0.7

if len(sys.argv) == 1:
    ltd = input("Limited horizons, (Y)ay/(N)ay")
    while ltd.strip().lower() != "y" and ltd.strip().lower() != "n":
        ltd = input("Limited horizons, (Y)ay/(N)ay")
    if ltd.strip().lower() == "y":
        N_HORIZONS_TO_TEST = [ 7, 10, 15, 30, 45, 60, 75, 90]
    else:
        N_HORIZONS_TO_TEST = [1, 3, 5, 7, 10, 15, 30, 45, 60, 75, 90]
else:
    if sys.argv[1].strip().lower() == "ltd":
        print("limited horizons")
        N_HORIZONS_TO_TEST = [ 7, 10, 15, 30, 45, 60, 75, 90]
    elif sys.argv[1].strip().lower() == "all":
        print("all horizons")
        N_HORIZONS_TO_TEST = [1, 3, 5, 7, 10, 15, 30, 45, 60, 75, 90]
    else:
        print("Invalid argument. Please use 'ltd' or 'all'.")
        sys.exit(1)

CV_SPLITS = 10
TRANSACTION_COST = 0.005
# run is failed if MAPE or RMSPE > this value, or if Prophet fails to converge
FAILURE_THRESHOLD = 10000

def get_feature_sets(df):
    base_price_cols = ['open', 'high', 'low', 'close', 'volume']
    tech_indicator_cols = [col for col in df.columns if col.startswith(('sma_', 'ema_', 'rocr_', 'bb_', 'rsi_', 'macd'))]
    baseline_features = base_price_cols + tech_indicator_cols
    baseline_features = [f for f in baseline_features if f in df.columns]
    all_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return {
        "Baseline_Features": baseline_features,
        "All_Features": all_features
    }

# backtesting function
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

    # calculate metrics
    final_value = df['portfolio_value'].iloc[-1]
    total_profit = final_value - initial_balance
    roi = (total_profit / initial_balance) if initial_balance != 0 else 0

    mean_daily_return = df['daily_return'].mean()
    std_daily_return = df['daily_return'].std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(365) if std_daily_return != 0 else 0

    annualized_return = roi
    cumulative_max = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0

    metrics = {
        'Final Balance': final_value,
        'Total Profit ($)': total_profit,
        'Return on Investment (%)': roi * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown (%)': abs(max_drawdown) * 100
    }
    return metrics, df

def optimize_prediction_horizon(df, features, model_type):
    print(f"Optimizing '{model_type}' prediction horizon n")
    best_n = -1
    lowest_rmse = float('inf')

    min_val_size = len(df) // (CV_SPLITS + 1)
    horizons_to_test = [n for n in N_HORIZONS_TO_TEST if n <=(len(df)/(1-TRAIN_SIZE)*TRAIN_SIZE)*0.4]
    if not horizons_to_test:
        print(f"all N_HORIZONS_TO_TEST too large for the {min_val_size} val set size")
        return N_HORIZONS_TO_TEST[-1]

    for n in horizons_to_test:
        df_n = df.copy()
        df_n['target'] = df_n['close'].shift(-n)
        df_n.dropna(subset=['target'], inplace=True)
        X = df_n[features]
        y = df_n['target']
        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        fold_rmses = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            if model_type == 'xgboost':
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, early_stopping_rounds=25, eval_metric="rmse", random_state=42)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
            elif model_type == 'prophet':
                prophet_features = [f for f in features if f not in ['open', 'high', 'low', 'close', 'volume']]
                train_df = pd.DataFrame({'ds': X_train.index, 'y': y_train.values})
                for feature in prophet_features:
                    train_df[feature] = X_train[feature].values

                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_prior_scale = 10.0,
                holidays_prior_scale = 10.0,
                changepoint_prior_scale = 0.05)
                for feature in prophet_features:
                    model.add_regressor(feature)
                # get convergence issues
                original_stderr = sys.stderr
                sys.stderr = captured_stderr = StringIO()
                try:
                    model.fit(train_df, algorithm='LBFGS')
                finally:
                    sys.stderr = original_stderr

                log_output = captured_stderr.getvalue()
                if "Optimization terminated abnormally" in log_output:
                    print(f"Prophet convergence warning for n={n} during optimization.")

                future_df = pd.DataFrame({'ds': X_val.index})
                for feature in prophet_features:
                    future_df[feature] = X_val[feature].values
                forecast = model.predict(future_df)
                preds = forecast['yhat'].values

            fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_rmses.append(fold_rmse)

        avg_rmse = np.mean(fold_rmses)
        if avg_rmse < lowest_rmse:
            lowest_rmse = avg_rmse
            best_n = n

    print(f"-Found best n={best_n} for {model_type} with average RMSE: {lowest_rmse:.4f}")
    return best_n

def run_full_experiment():
    all_coin_results = {}
    all_predictions_for_plotting = []
    all_feature_importances = defaultdict(list)
    all_daily_returns = defaultdict(lambda: defaultdict(lambda: defaultdict(pd.Series)))
    run_status = defaultdict(lambda: defaultdict(list)) # track failures

    for coin in COINS:
        print("-" * 8 + f"Running Experiment for {coin}"+ "-" * 8)
        try:
            feature_filename = f"{coin.upper()}_features.feather"
            feature_path = os.path.join(FEATURE_DATA_DIR, feature_filename)
            df = pd.read_feather(feature_path)
            if 'date' in df.columns: df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df = df.sort_index()

            feature_sets = get_feature_sets(df)

            for feature_set_name, features in feature_sets.items():
                print(f"{feature_set_name} ({len(features)} features)")

                split_index = int(len(df) * TRAIN_SIZE)
                train_val_df = df.iloc[:split_index]
                test_df = df.iloc[split_index:]
                print(f"Train/Val set size {len(train_val_df)}, Test set size {len(test_df)}")

                current_features_df = train_val_df[features]
                best_n_xgb = optimize_prediction_horizon(current_features_df.copy(), features, 'xgboost')
                best_n_prophet = optimize_prediction_horizon(current_features_df, features, 'prophet')

                run_predictions = {'xgboost': None, 'prophet': None}

                results = {}
                for model_type, best_n in [('xgboost', best_n_xgb), ('prophet', best_n_prophet)]:
                    print(f"-Evaluating {model_type} on hold-out set with n={best_n}...")
                    df_n = df.copy()
                    df_n['target'] = df_n['close'].shift(-best_n)
                    df_n.dropna(subset=['target'], inplace=True)
                    X = df_n[features]
                    y = df_n['target']
                    assert df_n.index.is_monotonic_increasing
                    train_val_X = X[X.index < test_df.index[0]]
                    train_val_y = y[y.index < test_df.index[0]]
                    test_X = X[X.index >= test_df.index[0]]
                    test_y = y[y.index >= test_df.index[0]]

                    convergence_failure = False
                    if model_type == 'xgboost':
                        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, early_stopping_rounds=50, eval_metric="rmse", random_state=42)
                        model.fit(train_val_X, train_val_y, eval_set=[(test_X, test_y)], verbose=False)
                        predictions = model.predict(test_X)
                        importances = model.feature_importances_
                        feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
                        all_feature_importances[feature_set_name].append(feature_importance_df)

                    elif model_type == 'prophet':
                        prophet_features = [f for f in features if f not in ['open', 'high', 'low', 'close', 'volume']]
                        train_df = pd.DataFrame({'ds': train_val_X.index, 'y': train_val_y.values})
                        for feature in prophet_features: train_df[feature] = train_val_X[feature].values
                        model = Prophet()
                        for feature in prophet_features: model.add_regressor(feature)

                        original_stderr = sys.stderr
                        sys.stderr = captured_stderr = StringIO()
                        try:
                            model.fit(train_df, algorithm='LBFGS')
                        finally:
                            sys.stderr = original_stderr
                        log_output = captured_stderr.getvalue()
                        if "Optimization terminated abnormally" in log_output:
                            print(f"proptet convergence failure for {coin} with {feature_set_name}")
                            convergence_failure = True

                        future_df = pd.DataFrame({'ds': test_X.index})
                        for feature in prophet_features: future_df[feature] = test_X[feature].values
                        forecast = model.predict(future_df)
                        predictions = forecast['yhat'].values

                    rmse = np.sqrt(mean_squared_error(test_y, predictions))
                    mae = mean_absolute_error(test_y, predictions)

                    true_values = test_y.values
                    non_zero_mask = true_values != 0
                    relative_errors = (predictions[non_zero_mask] - true_values[non_zero_mask]) / true_values[non_zero_mask]
                    mape = np.mean(np.abs(relative_errors)) * 100 if len(relative_errors) > 0 else 0.0
                    rmspe = np.sqrt(np.mean(np.square(relative_errors))) * 100 if len(relative_errors) > 0 else 0.0

                    is_failure = False
                    if convergence_failure:
                        is_failure = True
                    if not np.isfinite(mape) or not np.isfinite(rmspe) or mape > FAILURE_THRESHOLD or rmspe > FAILURE_THRESHOLD:
                        is_failure = True
                        print(f"metrics just exploded MAPE={mape:.2f}%, RMSPE={rmspe:.2f}%")
                    run_status[feature_set_name][model_type].append({'coin': coin, 'failed': is_failure})

                    backtest_df = test_df.loc[test_y.index].copy()
                    backtest_df['prediction'] = predictions
                    run_predictions[model_type] = backtest_df[['close', 'prediction']]

                    backtest_df['signal'] = (backtest_df['prediction'] / backtest_df['close']) - 1

                    profit_metrics, backtest_df_with_returns = analyze_profit(backtest_df, 'signal', transaction_cost=TRANSACTION_COST, risk_fraction=1.0)
                    all_daily_returns[feature_set_name][model_type][coin] = backtest_df_with_returns['daily_return']

                    results[model_type] = {
                        'Best n': best_n,
                        'Test RMSE': rmse,
                        'Test MAE': mae,
                        'Test MAPE (%)': mape,
                        'Test RMSPE (%)': rmspe,
                        'is_failure': is_failure,
                        **profit_metrics
                    }

                buyhold_df = test_df.loc[test_y.index].copy()
                buyhold_df['hold_signal'] = 1.0
                buyhold_metrics, buyhold_df_with_returns = analyze_profit(
                    buyhold_df, 'hold_signal', transaction_cost=TRANSACTION_COST
                )
                results['buy_hold'] = buyhold_metrics
                all_daily_returns[feature_set_name]['buy_hold'][coin] = buyhold_df_with_returns['daily_return']

                # plot_df = run_predictions['xgboost'].rename(columns={'prediction': 'xgboost_pred'})
                # plot_df['prophet_pred'] = run_predictions['prophet']['prediction']
                # plot_df['coin'] = coin
                # plot_df['feature_set'] = feature_set_name
                # plot_df.reset_index(inplace=True)
                # if 'index' in plot_df.columns:
                #     plot_df.rename(columns={'index': 'date'}, inplace=True)
                # all_predictions_for_plotting.append({
                #     'df': plot_df,
                #     'n_xgb': best_n_xgb,
                #     'n_prophet': best_n_prophet
                # })

                print(f"Results for {coin} with {feature_set_name}")
                if coin not in all_coin_results:
                    all_coin_results[coin] = {}
                all_coin_results[coin][feature_set_name] = results
                for model_name, metrics in results.items():
                    print(f"Model: {model_name.upper()}")
                    for metric_name, value in metrics.items():
                        if metric_name == 'is_failure': continue
                        print(f"- {metric_name}: {value:.4f}" if isinstance(value, float) else f"- {metric_name}: {value}")
                print("\n")

        except Exception as e:
            print(f"An error occurred while processing {coin}: {e}")
            import traceback
            traceback.print_exc()

    return all_coin_results, all_predictions_for_plotting, all_feature_importances, all_daily_returns, run_status

def block_bootstrap(data, block_size=10, n_bootstraps=5000):
    np.random.seed(42)
    n = len(data)
    if n < block_size:
        return np.array([np.nan])

    bootstrap_means = np.empty(n_bootstraps)
    block_starts = np.arange(n - block_size + 1)

    for i in range(n_bootstraps):
        # sample block start indices
        resampled_block_starts = np.random.choice(block_starts, size=(n // block_size) + 1, replace=True)

        # concat blocks to form the bootstrap sample
        bootstrap_sample = np.concatenate([data[start:start+block_size] for start in resampled_block_starts])
        bootstrap_sample = bootstrap_sample[:n]

        bootstrap_means[i] = np.mean(bootstrap_sample)

    return bootstrap_means

def perform_statistical_tests(cross_asset_metrics):
    print(f"\n\n{'='*20} STATISTICAL SIGNIFICANCE TESTS {'='*20}")
    print("(Wilcoxon signed-rank test (non-parametric))")

    results = []
    metrics_to_test = ['Return on Investment (%)', 'Sharpe Ratio', 'Calmar Ratio']

    for feature_set, models in cross_asset_metrics.items():
        print(f"\n--- Feature Set: {feature_set} ---")
        model_names = list(models.keys())

        if 'buy_hold' not in model_names:
            print("'buy_hold' metrics not found, skipping statistical tests for this set.")
            continue

        for model_name in [m for m in model_names if m != 'buy_hold']:
            print(f"\n  Comparison: {model_name.upper()} vs. BUY & HOLD")
            for metric in metrics_to_test:
                model_scores = models[model_name].get(metric)
                buy_hold_scores = models['buy_hold'].get(metric)

                if model_scores and buy_hold_scores and len(model_scores) == len(buy_hold_scores):
                    try:
                        if len(model_scores) > 5 and np.std(np.array(model_scores) - np.array(buy_hold_scores)) > 0:
                            stat, p_value = wilcoxon(model_scores, buy_hold_scores)
                            print(f"- {metric}: p-value = {p_value:.4f}")
                            results.append({
                                'test_type': 'wilcoxon',
                                'feature_set': feature_set,
                                'comparison': f"{model_name.upper()} vs. BUY & HOLD",
                                'metric': metric,
                                'p_value': p_value,
                                'statistic': stat
                            })
                        else:
                            print(f"- {metric}: Not enough data or variance for a meaningful test.")
                    except Exception as e:
                        print(f"- {metric}: Could not perform test. Reason: {e}")
                else:
                    print(f"- {metric}: Data mismatch or missing, cannot compare.")
        if 'xgboost' in model_names and 'prophet' in model_names:
            print(f"\n  Comparison: XGBOOST vs. PROPHET")
            for metric in metrics_to_test:
                xgb_scores = models['xgboost'].get(metric)
                prophet_scores = models['prophet'].get(metric)

                if xgb_scores and prophet_scores and len(xgb_scores) == len(prophet_scores):
                    try:
                        if len(xgb_scores) > 5 and np.std(np.array(xgb_scores) - np.array(prophet_scores)) > 0:
                            stat, p_value = wilcoxon(xgb_scores, prophet_scores)
                            print(f"- {metric}: p-value = {p_value:.4f}")
                            results.append({
                                'test_type': 'wilcoxon',
                                'feature_set': feature_set,
                                'comparison': "XGBOOST vs. PROPHET",
                                'metric': metric,
                                'p_value': p_value,
                                'statistic': stat
                            })
                        else:
                            print(f"- {metric}: Not enough data or variance for a meaningful test.")
                    except Exception as e:
                        print(f"- {metric} - error: {e}")
                else:
                    print(f"- {metric}: Data mismatch or missing.")
    return results


def perform_bootstrap_analysis(all_daily_returns):
    print(f"\n\n{'='*20} BOOTSTRAP CIs {'='*20}")

    results = []

    for feature_set, models_data in all_daily_returns.items():
        print(f"\nSet: {feature_set}")

        buy_hold_returns_df = pd.DataFrame(models_data.get('buy_hold', {}))
        if buy_hold_returns_df.empty:
            print("Buy-Hold returns not available, skipping.")
        else:
            for model_name in [m for m in models_data if m != 'buy_hold']:
                model_returns_df = pd.DataFrame(models_data[model_name])
                if model_returns_df.empty:
                    continue
                aligned_model, aligned_bh = model_returns_df.align(buy_hold_returns_df, join='inner', axis=0)
                daily_diffs = aligned_model - aligned_bh
                cross_sectional_mean_diffs = daily_diffs.mean(axis=1).dropna()

                if len(cross_sectional_mean_diffs) < 20:
                    print(f"Model: {model_name.upper()} vs. BUY & HOLD")
                    print("Not enough overlapping data points for bootstrap analysis.")
                    continue
                bootstrap_distribution = block_bootstrap(cross_sectional_mean_diffs.values)
                ci_lower = np.percentile(bootstrap_distribution, 2.5)
                ci_upper = np.percentile(bootstrap_distribution, 97.5)
                mean_effect = np.mean(cross_sectional_mean_diffs)
                print(f"Model: {model_name.upper()} vs. BUY & HOLD")
                print(f"- Mean Daily Excess Return: {mean_effect: .6f}")
                print(f"- 95% Confidence Interval:  [{ci_lower: .6f}, {ci_upper: .6f}]")
                results.append({
                    'test_type': 'bootstrap',
                    'feature_set': feature_set,
                    'comparison': f"{model_name.upper()} vs. BUY & HOLD",
                    'statistic': 'Mean Daily Excess Return',
                    'value': mean_effect,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
        if 'xgboost' in models_data and 'prophet' in models_data:
            xgb_returns_df = pd.DataFrame(models_data['xgboost'])
            prophet_returns_df = pd.DataFrame(models_data['prophet'])

            if not xgb_returns_df.empty and not prophet_returns_df.empty:
                aligned_xgb, aligned_prophet = xgb_returns_df.align(prophet_returns_df, join='inner', axis=0)
                daily_diffs_models = aligned_xgb - aligned_prophet
                cross_sectional_mean_diffs_models = daily_diffs_models.mean(axis=1).dropna()

                if len(cross_sectional_mean_diffs_models) < 20:
                    print(f"Model: XGBOOST vs. PROPHET")
                    print("- Not enough overlapping data points for bootstrap analysis.")
                else:
                    bootstrap_distribution_models = block_bootstrap(cross_sectional_mean_diffs_models.values)
                    ci_lower_models = np.percentile(bootstrap_distribution_models, 2.5)
                    ci_upper_models = np.percentile(bootstrap_distribution_models, 97.5)
                    mean_effect_models = np.mean(cross_sectional_mean_diffs_models)
                    print(f"Model: XGBOOST vs. PROPHET")
                    print(f"- Mean Daily Return Difference (XGB - Prophet): {mean_effect_models: .6f}")
                    print(f"- 95% Confidence Interval:                      [{ci_lower_models: .6f}, {ci_upper_models: .6f}]")
                    results.append({
                        'test_type': 'bootstrap',
                        'feature_set': feature_set,
                        'comparison': 'XGBOOST vs. PROPHET',
                        'statistic': 'Mean Daily Return Difference (XGB - Prophet)',
                        'value': mean_effect_models,
                        'ci_lower': ci_lower_models,
                        'ci_upper': ci_upper_models
                    })
            else:
                print("Model: XGBOOST vs. PROPHET")
                print("- Return data for XGBoost or Prophet is missing, skipping comparison.")
    return results


if __name__ == "__main__":
    results, predictions, feature_importances, daily_returns, run_status = run_full_experiment()
    cross_asset_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for coin_data in results.values():
        for feature_set_name, models in coin_data.items():
            for model_name, metrics in models.items():
                for metric_name, value in metrics.items():
                    cross_asset_metrics[feature_set_name][model_name][metric_name].append(value)
    output_dir = "csv_results"
    os.makedirs(output_dir, exist_ok=True)
    horizons_str = "_".join(map(str, N_HORIZONS_TO_TEST))
    mean_results_data = []

    print(f"\n\n{'='*20} SUMMARY {'='*20}")

    for feature_set_name, models in cross_asset_metrics.items():
        print(f"\n--- Feature Set: {feature_set_name} ---")
        model_order = [m for m in ['xgboost', 'prophet', 'buy_hold'] if m in models]

        for model_name in model_order:
            current_row = {'feature_set': feature_set_name, 'model': model_name, 'horizons': horizons_str}
            metrics = models[model_name]
            print(f"-Model: {model_name.upper()}")

            if model_name != 'buy_hold':
                status_list = run_status[feature_set_name][model_name]
                num_runs = len(status_list)
                num_failures = sum(s['failed'] for s in status_list)
                failure_rate = (num_failures / num_runs) * 100 if num_runs > 0 else 0
                current_row['failure_rate'] = f"{failure_rate:.2f}% ({num_failures}/{num_runs})"
                print(f"- Failure Rate: {failure_rate:.2f}% ({num_failures}/{num_runs})")
                if num_failures > 0:
                    failed_coins = [s['coin'] for s in status_list if s['failed']]
                    current_row['failed_assets'] = ', '.join(failed_coins)
                    print(f"- Failed assets: {', '.join(failed_coins)}")
                else:
                    current_row['failed_assets'] = ''
            else:
                current_row['failure_rate'] = 'N/A'
                current_row['failed_assets'] = ''


            metric_order = [
                'Best n', 'Test RMSE', 'Test MAE', 'Test MAPE (%)', 'Test RMSPE (%)',
                'Final Balance', 'Total Profit ($)', 'Return on Investment (%)',
                'Sharpe Ratio', 'Calmar Ratio', 'Max Drawdown (%)'
            ]
            metrics_to_split = ['Test RMSE', 'Test MAE', 'Test MAPE (%)', 'Test RMSPE (%)', 'Return on Investment (%)', 'Sharpe Ratio', 'Calmar Ratio', 'Final Balance', 'Total Profit ($)', 'Max Drawdown (%)','Best n']
            metrics_for_bh_comparison = ['Return on Investment (%)', 'Sharpe Ratio', 'Calmar Ratio', 'Final Balance', 'Total Profit ($)', 'Max Drawdown (%)']

            for metric_name in metric_order:
                if metric_name not in metrics:
                    continue
                values = metrics[metric_name]

                if metric_name in metrics_to_split and model_name != 'buy_hold':
                    status_list = run_status[feature_set_name][model_name]
                    failed_indices = {i for i, s in enumerate(status_list) if s['failed']}
                    all_vals = [v for v in values if np.isfinite(v)]
                    successful_vals = [v for i, v in enumerate(values) if i not in failed_indices and np.isfinite(v)]

                    buy_hold_all_vals, buy_hold_successful_vals = [], []
                    if metric_name in metrics_for_bh_comparison:
                        buy_hold_metric_values = cross_asset_metrics[feature_set_name].get('buy_hold', {}).get(metric_name, [])
                        if buy_hold_metric_values:
                            buy_hold_all_vals = [v for v in buy_hold_metric_values if np.isfinite(v)]
                            buy_hold_successful_vals = [v for i, v in enumerate(buy_hold_metric_values) if i not in failed_indices and np.isfinite(v)]

                    print(f"- {metric_name} (all runs):")
                    if all_vals:
                        current_row[f'{metric_name} (all runs) - Model Mean'] = np.mean(all_vals)
                        current_row[f'{metric_name} (all runs) - Model Median'] = np.median(all_vals)
                        print(f"-0- Model Mean:      {np.mean(all_vals):.4f}")
                        print(f"-0- Model Median:    {np.median(all_vals):.4f}")
                        if buy_hold_all_vals:
                            current_row[f'{metric_name} (all runs) - B&H Mean'] = np.mean(buy_hold_all_vals)
                            current_row[f'{metric_name} (all runs) - B&H Median'] = np.median(buy_hold_all_vals)
                            print(f"-0- B&H Mean:        {np.mean(buy_hold_all_vals):.4f}")
                            print(f"-0- B&H Median:      {np.median(buy_hold_all_vals):.4f}")
                        if metric_name == 'Return on Investment (%)':
                            returns_for_gmean = [1 + r/100 for r in all_vals]
                            if all(r > 0 for r in returns_for_gmean):
                                gmean_val = (gmean(returns_for_gmean) - 1) * 100
                                current_row[f'{metric_name} (all runs) - Model GMean'] = gmean_val
                                print(f"-0- Model GMean:     {gmean_val:.4f}%")
                            if buy_hold_all_vals:
                                bh_returns_for_gmean = [1 + r/100 for r in buy_hold_all_vals]
                                if all(r > 0 for r in bh_returns_for_gmean):
                                    bh_gmean_val = (gmean(bh_returns_for_gmean) - 1) * 100
                                    current_row[f'{metric_name} (all runs) - B&H GMean'] = bh_gmean_val
                                    print(f"-0- B&H GMean:       {bh_gmean_val:.4f}%")

                    if num_failures > 0:
                        print(f"- {metric_name} (successful runs):")
                        if successful_vals:
                            current_row[f'{metric_name} (successful runs) - Model Mean'] = np.mean(successful_vals)
                            current_row[f'{metric_name} (successful runs) - Model Median'] = np.median(successful_vals)
                            print(f"-0- Model Mean:      {np.mean(successful_vals):.4f}")
                            print(f"-0- Model Median:    {np.median(successful_vals):.4f}")
                            if buy_hold_successful_vals:
                                current_row[f'{metric_name} (successful runs) - B&H Mean'] = np.mean(buy_hold_successful_vals)
                                current_row[f'{metric_name} (successful runs) - B&H Median'] = np.median(buy_hold_successful_vals)
                                print(f"-0- B&H Mean:        {np.mean(buy_hold_successful_vals):.4f}")
                                print(f"-0- B&H Median:      {np.median(buy_hold_successful_vals):.4f}")

                            if metric_name == 'Return on Investment (%)':
                                returns_for_gmean = [1 + r/100 for r in successful_vals]
                                if all(r > 0 for r in returns_for_gmean):
                                    gmean_val = (gmean(returns_for_gmean) - 1) * 100
                                    current_row[f'{metric_name} (successful runs) - Model GMean'] = gmean_val
                                    print(f"-0- Model GMean:     {gmean_val:.4f}%")
                                if buy_hold_successful_vals:
                                    bh_returns_for_gmean = [1 + r/100 for r in buy_hold_successful_vals]
                                    if all(r > 0 for r in bh_returns_for_gmean):
                                        bh_gmean_val = (gmean(bh_returns_for_gmean) - 1) * 100
                                        current_row[f'{metric_name} (successful runs) - B&H GMean'] = bh_gmean_val
                                        print(f"-0- B&H GMean:       {bh_gmean_val:.4f}%")
                else:
                    finite_values = [v for v in values if np.isfinite(v)]
                    if not finite_values: continue
                    arithmetic_mean = np.mean(finite_values)
                    median = np.median(finite_values)

                    if model_name == 'buy_hold' and metric_name in metrics_to_split:
                        current_row[f'{metric_name} (all runs) - Model Mean'] = arithmetic_mean
                        current_row[f'{metric_name} (all runs) - Model Median'] = median
                        current_row[f'{metric_name} (all runs) - B&H Mean'] = arithmetic_mean
                        current_row[f'{metric_name} (all runs) - B&H Median'] = median
                        print(f"- {metric_name}:")
                        print(f"-0- Arithmetic Mean: {arithmetic_mean:.4f}")
                        print(f"-0- Median:          {median:.4f}")
                    else:
                        current_row[f'{metric_name} - Arithmetic Mean'] = arithmetic_mean
                        current_row[f'{metric_name} - Median'] = median
                        print(f"- {metric_name}:")
                        print(f"-0- Arithmetic Mean: {arithmetic_mean:.4f}")
                        print(f"-0- Median:          {median:.4f}")
            mean_results_data.append(current_row)

    mean_results_df = pd.DataFrame(mean_results_data)
    mean_results_path = os.path.join(output_dir, f"{horizons_str}_mean_results.csv")
    mean_results_df.to_csv(mean_results_path, index=False)
    print(f"\nResults saved to {mean_results_path}")
    wilcoxon_results = perform_statistical_tests(cross_asset_metrics)
    bootstrap_results = perform_bootstrap_analysis(daily_returns)
    
    stat_sig_df = pd.concat([pd.DataFrame(wilcoxon_results), pd.DataFrame(bootstrap_results)], ignore_index=True)
    stat_sig_path = os.path.join(output_dir, f"{horizons_str}_statistical_significance.csv")
    stat_sig_df.to_csv(stat_sig_path, index=False)
    print(f"Statistical significance results saved to {stat_sig_path}")

    print(f"\n\n{'='*20} FEATURE IMPORTANCE (XGBOOST) {'='*20}")
    for feature_set_name, importance_dfs in feature_importances.items():
        if not importance_dfs:
            print(f"\nNo data for: {feature_set_name}")
            continue
        combined_importances = pd.concat(importance_dfs)
        average_importances = combined_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)

        print(f"\nAvg Feature Importance for: {feature_set_name}")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(average_importances)
        
        if feature_set_name == "All_Features":
            filename = f"{horizons_str}_feature_importance_all.csv"
        elif feature_set_name == "Baseline_Features":
            filename = f"{horizons_str}_feature_importance_baseline.csv"
        else:
            filename = f"{horizons_str}_feature_importance_{feature_set_name}.csv"
        
        filepath = os.path.join(output_dir, filename)
        average_importances.to_csv(filepath)
        print(f"Feature importances for {feature_set_name} saved to {filepath}")
