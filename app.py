import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
st.set_page_config(
    page_title="QuantEco OS v6.0",
    layout="wide",
    page_icon="📈"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .signal-strong-buy  { color: #00c851; font-weight: 700; }
    .signal-weak-buy    { color: #ffbb33; font-weight: 700; }
    .signal-sell        { color: #ff4444; font-weight: 700; }
    .signal-hold        { color: #aaaaaa; font-weight: 700; }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 12px;
        color: #856404;
        margin-top: 12px;
    }
    .bug-fixed {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 12px;
        color: #0c5460;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in [
    ('watchlist', []),
    ('custom_scan_us', []),
    ('custom_scan_th', []),
    ('scan_results_df', None),
    ('scan_market', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

DEFAULT_US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
DEFAULT_TH_STOCKS = ["PTT.BK", "CPALL.BK", "ADVANC.BK", "KBANK.BK",
                     "SCB.BK", "TTB.BK", "KTB.BK", "LH.BK", "GULF.BK"]

# ==========================================
# 2. INDICATOR ENGINE (Bug-fixed v6)
# ==========================================
def calculate_indicators(df):
    """
    FIX 1: RSI ใช้ Wilder's EMA (alpha=1/14) แทน Simple Rolling Mean
    FIX 4: Volatility เก็บแบบ daily std (ใช้คู่กับ position sizing เท่านั้น)
    NEW  : MACD (12/26/9), Bollinger Bands (20, 2σ), Volume MA
    """
    df = df.copy()
    close = df['Close']

    # --- EMA Trend ---
    df['EMA50']  = close.ewm(span=50,  adjust=False).mean()
    df['EMA200'] = close.ewm(span=200, adjust=False).mean()

    # --- RSI — Wilder's Smoothing (FIX 1) ---
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- MACD (12 / 26 / 9) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    df['MACD_Cross_Up']   = (
        (df['MACD'] > df['MACD_Signal']) &
        (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    )
    df['MACD_Cross_Down'] = (
        (df['MACD'] < df['MACD_Signal']) &
        (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    )

    # --- Bollinger Bands (20, ±2σ) ---
    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df['BB_Upper'] = bb_mid + bb_std * 2
    df['BB_Lower'] = bb_mid - bb_std * 2
    df['BB_Mid']   = bb_mid
    band_width     = (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
    df['BB_PctB']     = (close - df['BB_Lower']) / band_width
    df['BB_Bandwidth'] = band_width / bb_mid   # Squeeze indicator

    # --- Volatility (Daily Std — for position sizing only) ---
    df['Pct_Change']  = close.pct_change()
    df['Rolling_Vol'] = df['Pct_Change'].rolling(20).std()

    # --- Volume MA (Volume Confirmation) ---
    if 'Volume' in df.columns:
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20'].replace(0, np.nan)
    else:
        df['Vol_MA20']  = np.nan
        df['Vol_Ratio'] = np.nan

    return df


# ==========================================
# 3. BACKTEST ENGINE (Multi-Factor + Trailing Stop)
# ==========================================
def run_realistic_backtest(df, initial_capital=100_000, fee_pct=0.002, risk_pct=0.02):
    """
    FIX 2: ไม่มี Survivorship Bias disclaimer (แสดงใน UI)
    FIX 3: except: pass → แสดง error จริง
    FIX 4: Volatility ไม่ต้อง annualize เพราะใช้เป็น stop_distance (daily price move × 2.5)
    FIX 5: Win Rate = 0 เมื่อไม่มี trade (ไม่ใช่ 50)
    NEW  : Trailing Stop Loss
    NEW  : Time Stop (max_hold_days)
    NEW  : MACD + BB + Volume เป็น filter
    NEW  : Market Regime filter (ราคาอยู่เหนือ EMA200)
    NEW  : Sharpe Ratio, Calmar Ratio
    """
    position        = 0
    total_trades    = 0
    winning_trades  = 0
    cash            = initial_capital
    buy_price       = 0.0
    trailing_high   = 0.0
    hold_days       = 0
    max_hold_days   = 120          # Time Stop
    equity_curve    = []
    daily_returns   = []
    prev_equity     = initial_capital

    for i in range(200, len(df)):
        row           = df.iloc[i]
        current_price = float(row['Close'])
        ema50         = float(row['EMA50'])
        ema200        = float(row['EMA200'])
        rsi           = float(row['RSI'])   if not pd.isna(row['RSI'])   else 50.0
        rolling_vol   = float(row['Rolling_Vol']) if not pd.isna(row['Rolling_Vol']) else 0.02
        macd_val      = float(row['MACD'])        if not pd.isna(row['MACD'])        else 0.0
        macd_sig      = float(row['MACD_Signal']) if not pd.isna(row['MACD_Signal']) else 0.0
        bb_pctb       = float(row['BB_PctB'])     if not pd.isna(row['BB_PctB'])     else 0.5
        vol_ratio     = float(row['Vol_Ratio'])   if not pd.isna(row['Vol_Ratio'])   else 1.0

        current_equity = cash + (position * current_price if position > 0 else 0)

        # ── Market Regime Filter: ราคาต้องอยู่เหนือ EMA200 ────────────────
        regime_ok  = current_price > ema200

        # ── BUY (Multi-Factor) ────────────────────────────────────────────
        trend_ok   = ema50 > ema200
        rsi_ok     = 30 < rsi < 70
        macd_ok    = macd_val > macd_sig
        bb_ok      = bb_pctb < 0.90            # ไม่ซื้อใกล้ BB Upper
        vol_ok     = vol_ratio >= 1.0          # Volume ≥ ค่าเฉลี่ย (confirm)

        if regime_ok and trend_ok and rsi_ok and macd_ok and bb_ok and vol_ok and position == 0:
            risk_amount   = current_equity * risk_pct
            stop_distance = rolling_vol * current_price * 2.5
            stop_distance = max(stop_distance, current_price * 0.02)  # min 2%

            cost_price    = current_price * (1 + fee_pct)
            shares        = int(risk_amount / stop_distance)
            total_cost    = shares * cost_price

            if total_cost > cash:
                shares     = int(cash / cost_price)
                total_cost = shares * cost_price

            if shares > 0:
                position      = shares
                buy_price     = cost_price
                trailing_high = current_price
                hold_days     = 0
                cash         -= total_cost
                total_trades += 1

        # ── UPDATE Trailing Stop ──────────────────────────────────────────
        if position > 0:
            trailing_high  = max(trailing_high, current_price)
            trailing_stop  = trailing_high * 0.93   # 7% trailing
            hold_days     += 1

            # ── SELL (Multi-Factor Exit) ──────────────────────────────────
            trend_exit   = ema50 < ema200
            rsi_exit     = rsi > 78
            macd_exit    = macd_val < macd_sig
            bb_exit      = bb_pctb > 0.97           # Take Profit ที่ BB Upper
            trail_exit   = current_price < trailing_stop
            time_exit    = hold_days >= max_hold_days

            if trend_exit or rsi_exit or (macd_exit and trail_exit) or bb_exit or time_exit:
                sell_price = current_price * (1 - fee_pct)
                cash      += position * sell_price
                if sell_price > buy_price:
                    winning_trades += 1
                position      = 0
                trailing_high = 0.0
                hold_days     = 0

        current_equity = cash + (position * current_price if position > 0 else 0)
        daily_ret      = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        equity_curve.append(current_equity)
        daily_returns.append(daily_ret)
        prev_equity = current_equity

    final_equity   = cash + (position * float(df['Close'].iloc[-1]) if position > 0 else 0)
    net_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    # FIX 5: Win Rate = 0 เมื่อไม่มี trade
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

    max_drawdown = 0.0
    if equity_curve:
        eq_series    = pd.Series(equity_curve)
        peak         = eq_series.cummax()
        drawdown     = (eq_series - peak) / peak
        max_drawdown = drawdown.min() * 100

    # --- Sharpe Ratio (annualized, rf=0) ---
    sharpe = 0.0
    if daily_returns:
        dr = pd.Series(daily_returns)
        if dr.std() > 0:
            sharpe = (dr.mean() / dr.std()) * np.sqrt(252)

    # --- Calmar Ratio ---
    calmar = 0.0
    if max_drawdown < 0:
        calmar = net_return_pct / abs(max_drawdown)

    return net_return_pct, win_rate, max_drawdown, total_trades, sharpe, calmar


# ==========================================
# 4. SIGNAL ENGINE (Multi-Factor v6)
# ==========================================
def evaluate_stock_v6(info, df, is_thai_market=False):
    sector      = info.get('sector', 'Financial Services' if is_thai_market else 'Technology')
    last        = df.iloc[-1]
    close       = float(last['Close'])
    ema50       = float(last['EMA50'])
    ema200      = float(last['EMA200'])
    rsi         = float(last['RSI'])     if not pd.isna(last['RSI'])     else 50.0
    macd_val    = float(last['MACD'])    if not pd.isna(last['MACD'])    else 0.0
    macd_sig    = float(last['MACD_Signal']) if not pd.isna(last['MACD_Signal']) else 0.0
    bb_pctb     = float(last['BB_PctB']) if not pd.isna(last['BB_PctB']) else 0.5
    bb_bw       = float(last['BB_Bandwidth']) if not pd.isna(last['BB_Bandwidth']) else 0.1
    vol_ratio   = float(last['Vol_Ratio']) if not pd.isna(last['Vol_Ratio']) else 1.0
    vol_daily   = float(last['Rolling_Vol']) if not pd.isna(last['Rolling_Vol']) else 0.02

    is_uptrend   = ema50 > ema200
    regime_ok    = close > ema200
    macd_bull    = macd_val > macd_sig
    vol_confirm  = vol_ratio >= 1.2
    bb_squeeze   = bb_bw < float(df['BB_Bandwidth'].quantile(0.20))

    volatility    = vol_daily * close
    stop_loss     = close - (volatility * 2.5)
    stop_loss     = max(stop_loss, close * 0.93)
    risk_distance = close - stop_loss
    target_price  = close + (risk_distance * 2.5)   # R:R = 1:2.5
    entry_price   = close if rsi < 60 else ema50

    # --- Dividend yield สำหรับหุ้นไทย ---
    div_yield = info.get('dividendYield', 0) or 0
    div_str   = f" | ปันผล {div_yield*100:.1f}%" if is_thai_market and div_yield > 0 else ""

    # --- คะแนน Signal (0–5) ---
    score = sum([
        is_uptrend,
        regime_ok,
        macd_bull,
        30 < rsi < 65,
        vol_confirm,
    ])

    if score >= 4 and bb_pctb < 0.85:
        sig_code = "STRONG_BUY"
        sig_txt  = f"🟢 ซื้อสะสม (Strong){div_str}"
    elif score >= 3:
        sig_code = "WEAK_BUY"
        sig_txt  = f"🟡 ซื้อบางส่วน{div_str}"
    elif not regime_ok or rsi > 75 or (not macd_bull and not is_uptrend):
        sig_code = "SELL"
        sig_txt  = "🔴 ขาย/หลีกเลี่ยง"
        entry_price = target_price = stop_loss = "-"
    elif bb_squeeze:
        sig_code = "WATCH"
        sig_txt  = "👀 จับตา (Squeeze)"
    else:
        sig_code = "HOLD"
        sig_txt  = "⚪ ถือดูแนวโน้ม"
        entry_price = target_price = stop_loss = "-"

    return sig_code, sig_txt, entry_price, target_price, stop_loss, sector, score


# ==========================================
# 5. CHART: Interactive Candlestick + Indicators
# ==========================================
def plot_stock_chart(df, ticker):
    df_plot = df.tail(252).copy()   # 1 ปีล่าสุด

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.18, 0.17, 0.15],
        subplot_titles=[
            f"{ticker} — Price + EMA + Bollinger Bands",
            "Volume",
            "RSI (14)",
            "MACD (12/26/9)"
        ]
    )

    # ── Candlestick ───────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'],   close=df_plot['Close'],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
    ), row=1, col=1)

    # BB
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Upper'],
        line=dict(color='rgba(100,149,237,0.6)', width=1, dash='dot'),
        name='BB Upper', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Lower'],
        line=dict(color='rgba(100,149,237,0.6)', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(100,149,237,0.05)',
        name='BB Band', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Mid'],
        line=dict(color='rgba(100,149,237,0.4)', width=1),
        name='BB Mid', showlegend=False), row=1, col=1)

    # EMA
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA50'],
        line=dict(color='#ffa726', width=1.5), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA200'],
        line=dict(color='#ef5350', width=1.5), name='EMA 200'), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────
    colors_vol = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df_plot['Close'], df_plot['Open'])]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'],
        marker_color=colors_vol, name='Volume', showlegend=False), row=2, col=1)
    if 'Vol_MA20' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Vol_MA20'],
            line=dict(color='#ffa726', width=1.2), name='Vol MA20',
            showlegend=False), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'],
        line=dict(color='#ab47bc', width=1.5), name='RSI', showlegend=False), row=3, col=1)
    for level, color in [(70, 'rgba(239,83,80,0.5)'), (30, 'rgba(38,166,154,0.5)')]:
        fig.add_hline(y=level, line_dash='dash', line_color=color,
                      line_width=1, row=3, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────
    colors_macd = ['#26a69a' if v >= 0 else '#ef5350'
                   for v in df_plot['MACD_Hist'].fillna(0)]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_Hist'],
        marker_color=colors_macd, name='MACD Hist', showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'],
        line=dict(color='#42a5f5', width=1.5), name='MACD', showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_Signal'],
        line=dict(color='#ff7043', width=1.5), name='Signal', showlegend=False), row=4, col=1)

    fig.update_layout(
        height=700,
        template='plotly_dark',
        paper_bgcolor='rgba(15,32,39,0.95)',
        plot_bgcolor='rgba(15,32,39,0.95)',
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02, x=0),
        font=dict(family='sans-serif', size=11),
    )
    return fig


# ==========================================
# 6. DASHBOARD UI
# ==========================================
st.markdown("""
<div class="main-header">
    <h2 style="margin:0;font-size:24px;">📈 QuantEco OS v6.0</h2>
    <p style="margin:4px 0 0;opacity:0.75;font-size:13px;">
        Multi-Factor Signal Engine · Wilder RSI · MACD · Bollinger Bands · Trailing Stop · Sharpe/Calmar Ratio
    </p>
</div>
""", unsafe_allow_html=True)

# Bug-fix notice
with st.expander("✅ จุดที่แก้ไขจากเวอร์ชัน v5.2 (คลิกดู)"):
    fixes = [
        "FIX 1: RSI ใช้ Wilder's EMA (ewm alpha=1/14) แทน Simple Rolling Mean — สัญญาณตรงมาตรฐานแล้ว",
        "FIX 2: เพิ่ม Survivorship Bias Disclaimer — ผลลัพธ์ backtest ไม่ได้รวมหุ้นที่ Delisted",
        "FIX 3: except: pass → except Exception as e — แสดง error จริงแทน silent fail",
        "FIX 4: Volatility ใช้ถูกต้องใน Position Sizing (daily stop-distance ไม่ต้อง annualize)",
        "FIX 5: Win Rate = 0 เมื่อ Total Trades = 0 (ไม่ใช่ default 50%)",
        "NEW: MACD (12/26/9) เป็น Momentum Confirmation Gate",
        "NEW: Bollinger Bands (20, 2σ) + %B + Bandwidth (Squeeze Detection)",
        "NEW: Volume Confirmation (Vol > MA20 × 1.0 ถึงจะ BUY)",
        "NEW: Market Regime Filter (ราคาต้องอยู่เหนือ EMA200)",
        "NEW: Trailing Stop Loss (7% จาก High) + Time Stop (120 วัน)",
        "NEW: Sharpe Ratio + Calmar Ratio ใน Backtest",
        "NEW: Dividend Yield รวมใน Signal (หุ้นไทย)",
        "NEW: Signal Score 0–5 → Strong Buy / Weak Buy / Watch / Hold / Sell",
        "NEW: Interactive Chart (Candlestick + BB + EMA + Volume + RSI + MACD)",
    ]
    for f in fixes:
        st.markdown(f'<div class="bug-fixed">✔ {f}</div>', unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "เลือกฟังก์ชันหลัก:",
    ["🔍 สแกนหุ้น & วางแผนเทรด", "📊 ดูกราฟเชิงลึก", "💼 พอร์ตจำลอง (Risk Manager)"]
)

# ==========================================
# PAGE 1: SCAN
# ==========================================
if menu == "🔍 สแกนหุ้น & วางแผนเทรด":
    st.subheader("🔍 สแกนหุ้นระดับสถาบัน")

    market_select = st.radio(
        "เลือกตลาดหุ้น:",
        ["หุ้นสหรัฐฯ (US Big Tech & Bluechips)", "หุ้นไทยปันผลแข็งแกร่ง (SET)"],
        horizontal=True
    )
    is_thai = "SET" in market_select

    base_stocks = (
        DEFAULT_TH_STOCKS + st.session_state.custom_scan_th if is_thai
        else DEFAULT_US_STOCKS + st.session_state.custom_scan_us
    )

    st.markdown("#### ➕ เพิ่มหุ้นเข้าตารางสแกน")
    c1, c2 = st.columns([3, 1])
    with c1:
        custom_ticker = st.text_input(
            "ชื่อย่อหุ้น (เช่น NVDA หรือ SCC):", key="scan_input"
        ).upper().strip()
    with c2:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        if st.button("เพิ่มเข้าตาราง"):
            if custom_ticker:
                ft = f"{custom_ticker}.BK" if is_thai and not custom_ticker.endswith(".BK") else custom_ticker
                lst = 'custom_scan_th' if is_thai else 'custom_scan_us'
                if ft not in st.session_state[lst]:
                    st.session_state[lst].append(ft)
                    st.rerun()

    col_scan, col_clear = st.columns([2, 1])
    with col_scan:
        do_scan = st.button("🚀 สแกน Multi-Factor (ดึงข้อมูล 10 ปี)", type="primary")
    with col_clear:
        if st.button("🧹 ล้างหุ้นที่เพิ่ม"):
            st.session_state.custom_scan_th = []
            st.session_state.custom_scan_us = []
            st.rerun()

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Survivorship Bias Disclaimer:</strong>
    รายการหุ้นเริ่มต้นเป็นหุ้นที่ยังอยู่รอดและ Outperform จนถึงปัจจุบัน
    ผลลัพธ์ Backtest จะดีกว่าความเป็นจริงเนื่องจากไม่รวมหุ้นที่ Delisted หรือล้มละลาย
    ใช้เพื่อเปรียบเทียบเชิงสัมพัทธ์เท่านั้น ไม่ใช่การันตีผลตอบแทน
    </div>
    """, unsafe_allow_html=True)

    if do_scan:
        results = []
        errors  = []
        prog    = st.progress(0)
        status  = st.empty()

        for idx, t in enumerate(base_stocks):
            status.text(f"กำลังดึงข้อมูล {t} ({idx+1}/{len(base_stocks)})...")
            prog.progress((idx + 1) / len(base_stocks))
            try:
                s = yf.Ticker(t)
                d = s.history(period="10y")
                if d.empty or len(d) < 200:
                    errors.append(f"{t}: ข้อมูลไม่เพียงพอ (ต้องการ 200+ วัน)")
                    continue

                d = calculate_indicators(d)
                net_ret, win_w, mdd, n_trades, sharpe, calmar = run_realistic_backtest(d)
                sig_code, sig_txt, ent, tg, sl_p, sector, score = evaluate_stock_v6(
                    s.info, d, is_thai_market=is_thai
                )

                # Dividend (หุ้นไทย)
                div_yield = s.info.get('dividendYield', 0) or 0

                def fmt(v):
                    return f"{v:,.2f}" if isinstance(v, float) else "-"

                results.append({
                    "หุ้น":            t.replace(".BK", ""),
                    "Sector":          sector,
                    "Signal":          sig_txt,
                    "Score":           f"{score}/5",
                    "Entry":           fmt(ent),
                    "Target":          fmt(tg),
                    "Stop Loss":       fmt(sl_p),
                    "Div Yield":       f"{div_yield*100:.1f}%" if div_yield > 0 else "-",
                    "Win Rate":        f"{win_w:.1f}%" if n_trades > 0 else "N/A",
                    "Trades (10y)":    str(n_trades),
                    "Net Return":      f"{net_ret:+.1f}%",
                    "Max Drawdown":    f"{mdd:.1f}%",
                    "Sharpe":          f"{sharpe:.2f}",
                    "Calmar":          f"{calmar:.2f}",
                })

            except Exception as e:
                errors.append(f"{t}: {str(e)[:80]}")

        prog.empty()
        status.empty()

        if errors:
            with st.expander(f"⚠️ ข้อผิดพลาด {len(errors)} รายการ (FIX 3: แสดง error จริง)"):
                for err in errors:
                    st.warning(err)

        if results:
            res_df = pd.DataFrame(results)
            st.session_state.scan_results_df = res_df
            st.session_state.scan_market     = is_thai

            st.success(f"✅ สแกนสำเร็จ {len(results)} หุ้น | Multi-Factor: EMA + RSI + MACD + BB + Volume")
            st.dataframe(res_df, use_container_width=True, height=420)

            # Summary counts
            c1, c2, c3, c4 = st.columns(4)
            strong_buy = sum(1 for r in results if "Strong" in r['Signal'])
            weak_buy   = sum(1 for r in results if "บางส่วน" in r['Signal'])
            watch      = sum(1 for r in results if "Squeeze" in r['Signal'])
            sell_cnt   = sum(1 for r in results if "ขาย" in r['Signal'])
            c1.metric("🟢 Strong Buy", strong_buy)
            c2.metric("🟡 Weak Buy",   weak_buy)
            c3.metric("👀 Watch",       watch)
            c4.metric("🔴 Sell/Avoid", sell_cnt)
        else:
            st.error("ไม่สามารถดึงข้อมูลได้ กรุณาลองอีกครั้ง")

    # Add to watchlist
    st.markdown("---")
    st.subheader("💼 บันทึกหุ้นเข้าพอร์ตจำลอง")
    cx, cy = st.columns([3, 1])
    with cx:
        add_ticker = st.text_input("ชื่อย่อหุ้น (เช่น AAPL หรือ PTT):", key="wl_in").upper().strip()
    with cy:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        if st.button("➕ บันทึกเข้าพอร์ต"):
            if add_ticker:
                ft = f"{add_ticker}.BK" if is_thai and not add_ticker.endswith(".BK") else add_ticker
                if ft not in st.session_state.watchlist:
                    st.session_state.watchlist.append(ft)
                    st.success(f"เพิ่ม {ft} แล้ว!")
                else:
                    st.warning("มีหุ้นนี้ในพอร์ตแล้ว")


# ==========================================
# PAGE 2: CHART
# ==========================================
elif menu == "📊 ดูกราฟเชิงลึก":
    st.subheader("📊 วิเคราะห์กราฟเชิงลึก")

    col_t, col_p, col_btn = st.columns([2, 1, 1])
    with col_t:
        chart_ticker = st.text_input(
            "ชื่อย่อหุ้น (เช่น AAPL, PTT.BK, NVDA):", value="AAPL"
        ).upper().strip()
    with col_p:
        period_map = {"1 ปี": "1y", "2 ปี": "2y", "5 ปี": "5y", "10 ปี": "10y"}
        period_sel = st.selectbox("ช่วงเวลา:", list(period_map.keys()), index=0)
    with col_btn:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        show_chart = st.button("📈 แสดงกราฟ", type="primary")

    if show_chart and chart_ticker:
        with st.spinner(f"กำลังดึงข้อมูล {chart_ticker}..."):
            try:
                s = yf.Ticker(chart_ticker)
                d = s.history(period=period_map[period_sel])
                if d.empty or len(d) < 200:
                    st.error("ข้อมูลไม่เพียงพอ กรุณาตรวจสอบชื่อหุ้น")
                else:
                    d = calculate_indicators(d)
                    is_th = chart_ticker.endswith(".BK")
                    sig_code, sig_txt, ent, tg, sl_p, sector, score = evaluate_stock_v6(
                        s.info, d, is_thai_market=is_th
                    )

                    # Metrics row
                    last_price = float(d['Close'].iloc[-1])
                    prev_price = float(d['Close'].iloc[-2])
                    chg_pct    = (last_price - prev_price) / prev_price * 100
                    last_rsi   = float(d['RSI'].iloc[-1]) if not pd.isna(d['RSI'].iloc[-1]) else 0
                    last_macd  = float(d['MACD'].iloc[-1]) if not pd.isna(d['MACD'].iloc[-1]) else 0
                    last_sig   = float(d['MACD_Signal'].iloc[-1]) if not pd.isna(d['MACD_Signal'].iloc[-1]) else 0

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("ราคาปัจจุบัน",    f"{last_price:,.2f}", f"{chg_pct:+.2f}%")
                    m2.metric("RSI (14)",        f"{last_rsi:.1f}")
                    m3.metric("MACD vs Signal",  f"{last_macd:.3f}", f"Signal: {last_sig:.3f}")
                    m4.metric("Signal Score",    f"{score}/5")
                    m5.metric("คำแนะนำ",         sig_txt[:10])

                    # Chart
                    fig = plot_stock_chart(d, chart_ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    # Detail panel
                    with st.expander("📋 รายละเอียด Entry / Target / Stop Loss"):
                        d1, d2, d3 = st.columns(3)
                        d1.metric("Entry Price",    f"{ent:,.2f}" if isinstance(ent, float) else "-")
                        d2.metric("Target (2.5R)",  f"{tg:,.2f}"  if isinstance(tg,  float) else "-")
                        d3.metric("Stop Loss (7%T)", f"{sl_p:,.2f}" if isinstance(sl_p, float) else "-")

                        if isinstance(ent, float) and isinstance(tg, float) and isinstance(sl_p, float):
                            rr = (tg - ent) / (ent - sl_p) if ent > sl_p else 0
                            st.info(f"Risk:Reward Ratio = **1 : {rr:.1f}**  |  Sector: {sector}")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")


# ==========================================
# PAGE 3: PORTFOLIO RISK MANAGER
# ==========================================
else:
    st.subheader("💼 Portfolio Risk Manager v6")

    if not st.session_state.watchlist:
        st.info("พอร์ตจำลองยังว่างอยู่ กรุณาเพิ่มหุ้นจากหน้าสแกนครับ")
    else:
        col_r, col_c = st.columns([1, 1])
        with col_r:
            if st.button("🔄 รีเฟรชข้อมูลพอร์ต"):
                st.rerun()
        with col_c:
            if st.button("🗑️ ล้างพอร์ตทั้งหมด"):
                st.session_state.watchlist = []
                st.rerun()

        wl_results    = []
        sector_counts = {}
        total_signals = {"STRONG_BUY": 0, "WEAK_BUY": 0, "HOLD": 0, "SELL": 0, "WATCH": 0}

        with st.spinner("กำลังวิเคราะห์ความเสี่ยงพอร์ต..."):
            for t in st.session_state.watchlist:
                try:
                    s = yf.Ticker(t)
                    d = s.history(period="2y")
                    if d.empty or len(d) < 50:
                        st.warning(f"{t}: ข้อมูลไม่เพียงพอ")
                        continue
                    d = calculate_indicators(d)
                    is_th = ".BK" in t
                    sig_code, sig_txt, _, _, _, sector, score = evaluate_stock_v6(
                        s.info, d, is_thai_market=is_th
                    )
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    total_signals[sig_code] = total_signals.get(sig_code, 0) + 1

                    last_price = float(d['Close'].iloc[-1])
                    chg        = float(d['Pct_Change'].iloc[-1]) * 100
                    last_rsi   = float(d['RSI'].iloc[-1]) if not pd.isna(d['RSI'].iloc[-1]) else 0
                    bb_pctb    = float(d['BB_PctB'].iloc[-1]) if not pd.isna(d['BB_PctB'].iloc[-1]) else 0.5

                    wl_results.append({
                        "หุ้น":              t.replace(".BK", ""),
                        "Sector":            sector,
                        "ราคา":              f"{last_price:,.2f}",
                        "เปลี่ยน (%)":       f"{chg:+.2f}%",
                        "RSI":               f"{last_rsi:.0f}",
                        "BB %B":             f"{bb_pctb:.2f}",
                        "Signal Score":      f"{score}/5",
                        "สถานะ":             sig_txt,
                    })

                except Exception as e:
                    st.warning(f"{t}: {str(e)[:60]}")

        # Portfolio Health Summary
        n = len(st.session_state.watchlist)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("หุ้นในพอร์ต",     n)
        c2.metric("🟢 Buy Signals",   total_signals.get("STRONG_BUY", 0) + total_signals.get("WEAK_BUY", 0))
        c3.metric("🔴 Sell Signals",  total_signals.get("SELL", 0))
        c4.metric("Sectors",          len(sector_counts))

        # Dynamic overweight threshold (FIX)
        threshold = max(2, int(n * 0.35))
        overweight = [s for s, c in sector_counts.items() if c > threshold and s != 'N/A']
        if overweight:
            st.error(f"⚠️ Sector Concentration Risk: กระจุกตัวใน {', '.join(overweight)} (เกิน {threshold} ตัว)")
        else:
            st.success("✅ การกระจาย Sector สมดุลดี")

        if wl_results:
            st.dataframe(pd.DataFrame(wl_results), use_container_width=True)

        # Sector Pie
        if sector_counts:
            fig_pie = go.Figure(go.Pie(
                labels=list(sector_counts.keys()),
                values=list(sector_counts.values()),
                hole=0.45,
                marker=dict(colors=[
                    '#42a5f5','#26a69a','#ffa726','#ef5350',
                    '#ab47bc','#66bb6a','#ff7043','#78909c','#ec407a'
                ])
            ))
            fig_pie.update_layout(
                title="Sector Allocation",
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
