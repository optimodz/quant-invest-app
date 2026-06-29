import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
st.set_page_config(page_title="QuantEco OS v8.0", layout="wide", page_icon="📡")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #000000, #1a237e, #000000); padding: 22px; border-radius: 12px; color: white; border-left: 6px solid #534bae; margin-bottom: 20px;}
    .stat-box { background: #111; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center; color: #fff;}
    .highlight { color: #534bae; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if 'watchlist' not in st.session_state: st.session_state.watchlist = []

# ==========================================
# 2. DYNAMIC UNIVERSE SCRAPING (The Secret Weapon)
# ==========================================
@st.cache_data(ttl=86400) # จำข้อมูลไว้ 1 วันเต็ม จะได้ไม่โหลดซ้ำซาก
def get_sp500_tickers():
    try:
        # สแกนลากอวนจาก Wikipedia สดๆ
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500 = tables[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in sp500]
    except:
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

@st.cache_data(ttl=86400)
def get_set50_tickers():
    # รายชื่อ SET50 หุ้นไทยที่แข็งแกร่งที่สุด
    return ["ADVANC.BK", "AOT.BK", "BDMS.BK", "BEM.BK", "BGRIM.BK", "BH.BK", "CPALL.BK", "CPN.BK", "CRC.BK", "DELTA.BK", "EA.BK", "GULF.BK", "INTUCH.BK", "KBANK.BK", "KTB.BK", "MINT.BK", "PTT.BK", "PTTEP.BK", "SCB.BK", "SCC.BK", "TISCO.BK", "TRUE.BK", "TTB.BK", "WHA.BK"]

@st.cache_data(ttl=3600)
def get_macro_regime(ticker):
    try:
        macro = yf.download(ticker, period="5y", progress=False)['Close']
        if isinstance(macro, pd.DataFrame): macro = macro.iloc[:, 0]
        ema200 = macro.ewm(span=200, adjust=False).mean()
        return (macro > ema200).squeeze()
    except:
        return None

# ==========================================
# 3. HIGH-SPEED QUANT ENGINE (Numpy + ATR)
# ==========================================
def calculate_indicators(df):
    close = df['Close']
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    df['EMA200'] = close.ewm(span=200, adjust=False).mean()

    # Wilder's RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    return df.dropna()

def run_fast_backtest(df, macro_series, initial_capital=100000, risk_pct=0.02):
    if macro_series is not None:
        df['Macro_OK'] = macro_series.reindex(df.index).ffill()
    else:
        df['Macro_OK'] = True

    c = df['Close'].to_numpy()
    o = df['Open'].to_numpy()
    ema50 = df['EMA50'].to_numpy()
    ema200 = df['EMA200'].to_numpy()
    rsi = df['RSI'].to_numpy()
    macd = df['MACD'].to_numpy()
    macd_sig = df['MACD_Sig'].to_numpy()
    atr = df['ATR'].to_numpy()
    macro_ok = df['Macro_OK'].to_numpy()
    
    pos, cash, trades, wins = 0, initial_capital, 0, 0
    buy_p, trail_h = 0.0, 0.0
    equity = np.zeros(len(c))
    
    for i in range(1, len(c) - 1):
        curr_p, next_o = c[i], o[i+1] # Execution Delay T+1
        curr_eq = cash + (pos * curr_p if pos > 0 else 0)
        equity[i] = curr_eq

        # BUY LOGIC
        if pos == 0 and macro_ok[i] and (ema50[i] > ema200[i]) and (30 < rsi[i] < 70) and (macd[i] > macd_sig[i]):
            risk_amt = curr_eq * risk_pct
            stop_dist = (atr[i] * 2.5) if atr[i] > 0 else curr_p * 0.05
            
            cost = next_o * 1.002 # + Fee
            shares = int(risk_amt / stop_dist) if stop_dist > 0 else 0
            if shares * cost > cash: shares = int(cash / cost)
                
            if shares > 0:
                pos, buy_p, trail_h = shares, cost, next_o
                cash -= (shares * cost)
                trades += 1
                
        # SELL LOGIC (Trailing Stop via ATR)
        elif pos > 0:
            trail_h = max(trail_h, curr_p)
            trail_stop = trail_h - (atr[i] * 3)
            
            if (ema50[i] < ema200[i]) or (rsi[i] > 75) or (curr_p < trail_stop):
                sell_p = next_o * 0.998 # - Fee
                if next_o < trail_stop: sell_p = next_o * 0.998 # Slippage on Gap Down
                
                cash += pos * sell_p
                if sell_p > buy_p: wins += 1
                pos = 0

    equity[-1] = cash + (pos * c[-1] if pos > 0 else 0)
    ret_pct = ((equity[-1] - initial_capital) / initial_capital) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0.0
    
    peak = np.maximum.accumulate(equity)
    mdd = np.min((equity - peak) / peak) * 100 if len(equity) > 0 else 0.0
    
    return ret_pct, win_rate, mdd, trades

# ==========================================
# 4. DASHBOARD & UI
# ==========================================
st.markdown("""<div class="main-header">
    <h1 style="margin:0;">📡 QuantEco OS v8.0 - Radar Edition</h1>
    <p style="margin:5px 0 0; color:#ddd;">ระบบสแกนหุ้นอัตโนมัติ ลากอวนข้อมูลจาก Wikipedia + วิเคราะห์ด้วยสมการ ATR แบบกองทุน</p>
</div>""", unsafe_allow_html=True)

menu = st.sidebar.selectbox("เมนูหลัก:", ["🔭 เรดาร์สแกนหุ้น (Dynamic Scan)", "💼 พอร์ตจำลอง"])

if menu == "🔭 เรดาร์สแกนหุ้น (Dynamic Scan)":
    market = st.radio("เลือกน่านน้ำที่จะสแกน:", ["🇺🇸 หุ้นอเมริกา (S&P 500)", "🇹🇭 หุ้นไทย (SET50)"], horizontal=True)
    is_thai = "SET50" in market
    
    # 🎯 พระเอกของงาน: ดึงข้อมูลสด!
    all_tickers = get_set50_tickers() if is_thai else get_sp500_tickers()
    macro_ticker = "^SET.BK" if is_thai else "SPY"
    
    st.info(f"ดึงข้อมูลหุ้นในลิสต์สำเร็จ **{len(all_tickers)}** ตัว! (Macro Index: {macro_ticker})")
    
    limit = st.slider("จำนวนหุ้นที่จะสแกน (ป้องกันโดนบล็อก IP):", 10, len(all_tickers), 20, 10)
    scan_list = all_tickers[:limit]
    
    if st.button("🚀 สแกนหาหุ้นแกร่งสุดในตลาดตอนนี้!", type="primary"):
        macro_series = get_macro_regime(macro_ticker)
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(scan_list):
            status_text.text(f"กำลังประมวลผล: {t} ({i+1}/{limit})")
            progress_bar.progress((i + 1) / limit)
            
            try:
                # โหลดประวัติ 5 ปีก็พอสำหรับเช็กเทรนด์ เพื่อความไว
                d = yf.download(t, period="5y", progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d = d.xs(t, axis=1, level=1)
                    
                if len(d) > 200:
                    d = calculate_indicators(d)
                    ret, win, mdd, n = run_fast_backtest(d, macro_series)
                    
                    # เช็ก Signal วันสุดท้าย
                    last = d.iloc[-1]
                    c, atr = float(last['Close']), float(last['ATR'])
                    up = float(last['EMA50']) > float(last['EMA200'])
                    m_ok = macro_series.iloc[-1] if macro_series is not None else True
                    
                    stop = c - (atr * 2.5)
                    tg = c + ((c - stop) * 2.5)
                    
                    if up and m_ok and float(last['RSI']) < 65 and float(last['MACD']) > float(last['MACD_Sig']):
                        sig = "🟢 STRONG BUY"
                    elif not m_ok:
                        sig = "🔴 MACRO DOWN (ห้ามเทรด)"
                        c, tg, stop = "-", "-", "-"
                    else:
                        sig = "⚪ HOLD / WAIT"
                        c, tg, stop = "-", "-", "-"
                        
                    results.append({
                        "Ticker": t, "Signal": sig,
                        "Entry": f"{c:,.2f}" if isinstance(c, float) else c,
                        "Target": f"{tg:,.2f}" if isinstance(tg, float) else tg,
                        "Stop Loss": f"{stop:,.2f}" if isinstance(stop, float) else stop,
                        "Win Rate": f"{win:.1f}%",
                        "Net Profit (5y)": f"{ret:+.1f}%", "MDD": f"{mdd:.1f}%"
                    })
            except: pass
            
        status_text.empty()
        progress_bar.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            st.success(f"สแกนเสร็จสิ้น! พบหุ้นที่มีสัญญาณเข้าซื้อตามเกณฑ์สถาบันดังนี้:")
            
            # แก้ไขบั๊ก Pandas เวอร์ชันใหม่ (ใช้ map แทน applymap)
            try:
                styled_df = df_res.style.map(lambda x: 'color: #00c851' if 'BUY' in str(x) else ('color: #ff4444' if 'DOWN' in str(x) else ''), subset=['Signal'])
            except AttributeError:
                # สำรองไว้เผื่อรันในเครื่องที่ใช้ Pandas เวอร์ชันเก่า
                styled_df = df_res.style.applymap(lambda x: 'color: #00c851' if 'BUY' in str(x) else ('color: #ff4444' if 'DOWN' in str(x) else ''), subset=['Signal'])
                
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูลผ่านเกณฑ์ หรือติด Rate Limit")
