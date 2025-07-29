# ✅ 文件名：data.py
def run_model():
    import jqdatasdk as jq
    import pandas as pd
    import collections
    from ta.trend import MACD
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.utils.class_weight import compute_sample_weight
    import matplotlib.pyplot as plt
    import xgboost as xgb
    import numpy as np

    jq.auth('17744657702', 'Bcptptptp1226')  # ✅ 可删掉账号密码后部署（避免泄露）

    stock_code = '300750.XSHE'
    start_date = '2024-04-12'
    end_date = '2025-04-19'

    df = jq.get_price(stock_code, start_date=start_date, end_date=end_date,
                      frequency='1d', fields=['open', 'close', 'high', 'low', 'volume'], skip_paused=True)

    df['macd'] = MACD(close=df['close']).macd()
    df['rsi_6'] = RSIIndicator(close=df['close'], window=6).rsi()
    so = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['kdj_d'] = so.stoch_signal()
    df['kdj_j'] = 3 * so.stoch() - 2 * so.stoch_signal()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma5_slope'] = df['ma5'].diff()
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    df['body_strength'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)

    feature_cols = [
        'macd', 'rsi_6', 'kdj_d', 'kdj_j',
        'volume', 'atr', 'obv',
    ]

    df_clean = df[feature_cols].dropna()
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=feature_cols, index=df_clean.index)
    df_scaled.to_csv('features.csv')

    df = df_scaled.copy()
    train_end_date = pd.to_datetime('2025-01-01')
    test_start_date = pd.to_datetime('2025-01-01')
    test_end_date = pd.to_datetime('2025-04-19')
    lookforward_days = 5

    close_price = jq.get_price(stock_code, start_date=df.index[0].strftime('%Y-%m-%d'),
                               end_date=df.index[-1].strftime('%Y-%m-%d'),
                               frequency='1d', fields=['close'], skip_paused=True)['close']

    future_return = close_price.shift(-lookforward_days) / close_price - 1
    quantile = pd.qcut(future_return, 5, labels=False, duplicates='drop')

    def map_label(q):
        if q <= 2:
            return 0
        elif q == 4:
            return 2
        else:
            return 1

    labels = quantile.map(map_label)
    df = df.loc[labels.dropna().index]
    labels = labels.dropna().loc[df.index]

    signal_list = []
    dates = df.index[(df.index >= test_start_date) & (df.index <= test_end_date)]

    for today in dates:
        train_data = df[df.index < today]
        train_label = labels.loc[train_data.index]
        if len(train_data) < 50:
            continue

        model = xgb.XGBClassifier(objective='multi:softprob', num_class=3,
                                  n_estimators=100, learning_rate=0.1, max_depth=3,
                                  eval_metric='mlogloss')
        model.fit(train_data, train_label)
        pred = model.predict(df.loc[[today]])[0]
        signal_list.append({'date': today.strftime('%Y-%m-%d'), 'signal': int(pred)})

    signal_df = pd.DataFrame(signal_list)
    signal_df.to_csv('strategy_signals_300750.csv', index=False)
