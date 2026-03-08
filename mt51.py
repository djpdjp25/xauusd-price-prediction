# =========================================================
# MT5 - UNIFIED MULTI-TIMEFRAME LSTM AUTO TRADER
# ONE TRADE PER 15M CANDLE
# FIXED RISK: $50 PER TRADE
# ATR-BASED SL / TP
# 15M ENTRY + 4H CONFIRMATION
# =========================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time, datetime, joblib
from tensorflow.keras import layers, models

# ================= SETTINGS =================
SYMBOL = "XAUUSD"

TF_ENTRY   = mt5.TIMEFRAME_M15
TF_CONFIRM = mt5.TIMEFRAME_H4

LOOKBACK = 60
BARS = 300

MAGIC = 777001
MAX_DAILY_LOSS = -3000.0

# ---- FIXED RISK ----
RISK_USD = 50.0

# ---- ATR ----
ATR_PERIOD = 14
ATR_SL_MULT = 1.2
ATR_TP_MULT = 2.5

MODEL_15M  = "lstm_15m.h5"
MODEL_4H   = "lstm_4h.h5"
SCALER_15M = "scaler_15m.pkl"
SCALER_4H  = "scaler_4h.pkl"

LAST_TRADE_CANDLE = None

# ================= LSTM MODEL =================
def build_lstm():
    model = models.Sequential([
        layers.Input(shape=(LOOKBACK,1)),
        layers.LSTM(256, return_sequences=True),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(1)
    ])
    return model

# ================= CONNECT MT5 =================
if not mt5.initialize():
    raise RuntimeError("❌ MT5 init failed")
print("✅ MT5 connected")

# ================= LOAD MODELS =================
model15 = build_lstm()
model4  = build_lstm()

model15.load_weights(MODEL_15M)
model4.load_weights(MODEL_4H)

model15.trainable = False
model4.trainable = False

sc15 = joblib.load(SCALER_15M)
sc4  = joblib.load(SCALER_4H)

print("✅ Models & scalers loaded")

# ================= DATA =================
def get_data(tf, bars):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, bars)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df[["open","high","low","close"]]

# ================= DAILY PNL =================
def daily_pnl():
    today = datetime.datetime.now().date()
    deals = mt5.history_deals_get(
        datetime.datetime(today.year, today.month, today.day),
        datetime.datetime.now()
    )
    if deals is None:
        return 0
    return sum(d.profit for d in deals if d.magic == MAGIC)

# ================= ATR =================
def compute_atr(df):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean().iloc[-1]

# ================= LOT SIZE (FIXED $50 RISK) =================
def lot_from_risk(atr):
    sl_distance = atr * ATR_SL_MULT
    if sl_distance <= 0:
        return 0.01

    lot = RISK_USD / (sl_distance * 100)
    return round(max(0.01, min(lot, 5.0)), 2)

# ================= SEND ORDER =================
def send_order(direction, lot, atr, candle_time):
    global LAST_TRADE_CANDLE

    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if direction == "BUY" else tick.bid

    sl_dist = atr * ATR_SL_MULT
    tp_dist = atr * ATR_TP_MULT

    if direction == "BUY":
        sl = price - sl_dist
        tp = price + tp_dist
        otype = mt5.ORDER_TYPE_BUY
    else:
        sl = price + sl_dist
        tp = price - tp_dist
        otype = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": otype,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "LSTM_FIXED_50USD",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        LAST_TRADE_CANDLE = candle_time
        print(f"✅ {direction} | lot={lot} | risk=$50")
    else:
        print("❌ Order failed:", result.retcode)

# ================= MAIN LOOP =================
print("🟢 LSTM MT5 AUTO TRADER RUNNING")

while True:

    df15 = get_data(TF_ENTRY, BARS)

    current_candle = df15.index[-1]
    closed_candle  = df15.index[-2]

    # ---- DAILY LOSS PROTECTION ----
    if daily_pnl() <= MAX_DAILY_LOSS:
        print("🛑 Max daily loss hit")
        break

    # ---- ONE TRADE PER CANDLE (HARD LOCK) ----
    if LAST_TRADE_CANDLE == closed_candle:
        time.sleep(60)
        continue

    # ---- 15M SIGNAL ----
    closes = df15["close"].values.reshape(-1,1)
    X15 = sc15.transform(closes)[-LOOKBACK:].reshape(1,LOOKBACK,1)

    pred15 = float(sc15.inverse_transform(
        [[model15.predict(X15, verbose=0)[0][0]]]
    )[0][0])

    last_price = float(df15["close"].iloc[-1])
    direction = "BUY" if pred15 > last_price else "SELL"

    # ---- 4H CONFIRMATION ----
    df4 = get_data(TF_CONFIRM, 200)
    closes4 = df4["close"].values.reshape(-1,1)
    X4 = sc4.transform(closes4)[-LOOKBACK:].reshape(1,LOOKBACK,1)

    pred4 = float(sc4.inverse_transform(
        [[model4.predict(X4, verbose=0)[0][0]]]
    )[0][0])

    last4 = float(df4["close"].iloc[-1])

    if direction == "BUY" and pred4 <= last4:
        time.sleep(60); continue
    if direction == "SELL" and pred4 >= last4:
        time.sleep(60); continue

    atr = compute_atr(df15)
    lot = lot_from_risk(atr)

    if lot > 0:
        send_order(direction, lot, atr, closed_candle)

    time.sleep(60)

mt5.shutdown()
