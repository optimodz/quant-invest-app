import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

# ปิดการแจ้งเตือนจุกจิก
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="QuantEco OS v8.2", layout="wide", page_icon="🐉")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #0A192F, #112240, #0A192F); padding: 25px; border-radius: 12px; color: #E6F1FF; border-left: 6px solid #FFD700; margin-bottom: 25px; }
    div.stButton > button:first-child { background-color: #0F52BA !important; color: white !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINES (Scraping & Indicators)
# ==========================================
@st.cache_data(ttl=86400)
def get_set50_tickers():
    return [
        "ADVANC.BK", "AOT.BK", "AWC.BK", "BBL.BK", "BDMS.BK", "BEM.BK", "BGRIM.BK", 
        "BH.BK", "BJC.BK", "BTS.BK", "CBG.BK", "CENTEL.BK", "CPALL.BK", "CPF.BK", 
        "CPN.BK", "CRC.BK", "DELTA.BK", "EA.BK", "EGCO.BK", "GLOBAL.BK", "GPSC.BK", 
        "GULF.BK", "HMPRO.BK", "INTUCH.BK", "IRPC.BK", "IVL.BK", "KBANK.BK", "KCE.BK", 
        "KTB.BK", "KTC.BK", "LH.BK", "MINT.BK", "OR.BK", "OSP.BK", "PTT.BK", "PTTEP.BK", 
        "PTTGC.BK", "RATCH.BK", "SCB.BK", "SCC.BK", "SCGP.BK", "TISCO.BK", "TLI.BK", 
        "TOP.BK", "TRUE.BK", "TTB.BK", "TU.BK", "VGI.BK", "WHA.BK"
    ]

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return [t.replace('.', '-') for t in tables[0]['Symbol'].tolist()]
    except: return ["AAPL", "MSFT", "NVDA", "GOOGL"]

def calculate_indicators(df):
    close = df['Close']
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    df['EMA200'] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    df['MACD'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['ATR'] = pd.concat([df['High']-df['Low'], np.abs(df['High']-close.shift()), np.abs(df['Low']-close.shift())], axis=1).max(axis=1).rolling(14).mean()
    return df.dropna()

def run_fast_backtest(df, macro_series, initial_capital=100000, risk_pct=0.03):
    df['Macro_OK'] = macro_series.reindex(df.index).ffill() if macro_series is not None else True
    c, o, ema50, ema200, rsi, macd, macd_sig, atr, m_ok = [df[col].to_numpy() for col in ['Close', 'Open', 'EMA50', 'EMA200', 'RSI', 'MACD', 'MACD_Sig', 'ATR', 'Macro_OK']]
    pos, cash, trades, wins = 0, initial_capital, 0, 0
    buy_p, trail_h, equity = 0.0, 0.0, np.full(len(c), initial_capital)
    
    for i in range(1, len(c) - 1):
        if pos == 0 and m_ok[i] and ema50[i] > ema200[i] and 30 < rsi[i] < 70 and macd[i] > macd_sig[i]:
            risk_amt = cash * risk_pct
            stop_dist = (atr[i] * 2.5) or (c[i] * 0.05)
            shares = int(risk_amt / stop_dist)
            if shares * (o[i+1]*1.002) <= cash:
                pos, buy_p, trail_h = shares, (o[i+1]*1.002), o[i+1]
                cash -= (shares * buy_p)
                trades += 1
        elif pos > 0:
            trail_h = max(trail_h, c[i])
            if ema50[i] < ema200[i] or rsi[i] > 75 or c[i] < (trail_h - atr[i] * 3):
                cash += pos * (o[i+1]*0.998)
                if (o[i+1]*0.998) > buy_p: wins += 1
                pos = 0
    equity[-1] = cash + (pos * c[-1] if pos > 0 else 0)
    ret = ((equity[-1] - initial_capital) / initial_capital) * 100
    return ret, (wins/trades*100 if trades > 0 else 0), trades

# ==========================================
# 3. MAIN DASHBOARD
# ==========================================
st.markdown('<div class="main-header"><h1>🐉 QuantEco OS v8.2 - Final Edition</h1></div>', unsafe_allow_html=True)

menu = st.sidebar.selectbox("เมนูหลัก:", ["🔭 เรดาร์สแกนหุ้น", "💼 พอร์ตจำลอง"])

if menu == "🔭 เรดาร์สแกนหุ้น":
    market = st.radio("เลือกน่านน้ำ:", ["🇺🇸 หุ้นอเมริกา (S&P 500)", "🇹🇭 หุ้นไทย (SET50)"], horizontal=True)
    is_thai = "SET50" in market
    all_tickers = get_set50_tickers() if is_thai else get_sp500_tickers()
    
    limit = st.slider("จำนวนหุ้นที่จะสแกน:", 1, len(all_tickers), min(20, len(all_tickers)))
    scan_list = all_tickers[:limit]
    
    if st.button("🚀 เริ่มต้นสแกน"):
        macro_series = yf.download("^SET.BK" if is_thai else "SPY", period="5y", progress=False)['Close'].ewm(span=200, adjust=False).mean() < yf.download("^SET.BK" if is_thai else "SPY", period="5y", progress=False)['Close']
        results = []
        for t in scan_list:
            try:
                d = calculate_indicators(yf.download(t, period="5y", progress=False))
                ret, win, n = run_fast_backtest(d, macro_series)
                last = d.iloc[-1]
                if float(last['EMA50']) > float(last['EMA200']) and float(last['RSI']) < 75:
                    results.append({"Ticker": t, "Signal": "🟢 BUY", "Net Profit (5y)": f"{ret:+.1f}%", "Win Rate": f"{win:.1f}%"})
            except: continue
        
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("ไม่พบหุ้นเข้าเกณฑ์ในขณะนี้")
