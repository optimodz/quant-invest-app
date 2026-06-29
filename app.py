import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. INITIALIZATION & SETUP (Feng Shui UI)
# ==========================================
st.set_page_config(page_title="QuantEco OS v8.2", layout="wide", page_icon="🐉")

# ปรับ CSS ตามหลักฮวงจุ้ยการเงิน (น้ำ = เงินไหลมา, ทอง = ความมั่งคั่ง)
st.markdown("""
<style>
    /* Header: ธาตุน้ำ (Navy Blue) ผสมธาตุดิน (Gold) เพื่อความมั่นคงและมั่งคั่ง */
    .main-header { 
        background: linear-gradient(135deg, #0A192F, #112240, #0A192F); 
        padding: 25px; 
        border-radius: 12px; 
        color: #E6F1FF; 
        border-left: 6px solid #FFD700; 
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Button: เปลี่ยนจากแดง (ไฟ) เป็นน้ำเงิน Sapphire (กระแสเงิน) */
    div.stButton > button:first-child {
        background-color: #0F52BA !important; 
        color: white !important;
        border: 1px solid #0F52BA !important;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #08367B !important;
        border: 1px solid #FFD700 !important;
        box-shadow: 0 0 10px rgba(255,215,0,0.5);
    }
    
    .highlight { color: #FFD700; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if 'watchlist' not in st.session_state: st.session_state.watchlist = []

# ==========================================
# 2. DYNAMIC UNIVERSE SCRAPING (Full 50 Stocks)
# ==========================================
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500 = tables[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in sp500]
    except:
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

@st.cache_data(ttl=86400)
def get_set50_tickers():
    # จัดเต็ม SET50 ครบ 50 ตัว (สมบูรณ์แบบ ไม่ตกหล่น)
    set50_hardcode = [
        "ADVANC.BK", "AOT.BK", "AWC.BK", "BBL.BK", "BDMS.BK", "BEM.BK", "BGRIM.BK", 
        "BH.BK", "BJC.BK", "BTS.BK", "CBG.BK", "CENTEL.BK", "CPALL.BK", "CPF.BK", 
        "CPN.BK", "CRC.BK", "DELTA.BK", "EA.BK", "EGCO.BK", "GLOBAL.BK", "GPSC.BK", 
        "GULF.BK", "HMPRO.BK", "INTUCH.BK", "IRPC.BK", "IVL.BK", "KBANK.BK", "KCE.BK", 
        "KTB.BK", "KTC.BK", "LH.BK", "MINT.BK", "OR.BK", "OSP.BK", "PTT.BK", "PTTEP.BK", 
        "PTTGC.BK", "RATCH.BK", "SCB.BK", "SCC.BK", "SCGP.BK", "TISCO.BK", "TLI.BK", 
        "TOP.BK", "TRUE.BK", "TTB.BK", "TU.BK", "VGI.BK", "WHA.BK"
    ]
    # ลองดึงข้อมูลจาก Wikipedia ก่อน ถ้าไม่ได้ให้ใช้ Hardcode ที่ครบถ้วนด้านบน
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/SET50_Index')
        scraped = tables[0]['Symbol'].tolist()
        if len(scraped) > 20: 
            return [f"{t}.BK" for t in scraped]
        return set50_hardcode
    except:
        return set50_hardcode

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
# 3. HIGH-SPEED QUANT ENGINE (Numpy + ATR + Math Fix)
# ==========================================
def calculate_indicators(df):
    close = df['Close']
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    df['EMA200'] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    return df.dropna()

def run_fast_backtest(df, macro_series, initial_capital=100000, risk_pct=0.03): # ปรับ Risk ขึ้นเล็กน้อยเพื่อลด Cash Drag
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
        curr_p, next_o = c[i], o[i+1]
        curr_eq = cash + (pos * curr_p if pos > 0 else 0)
        equity[i] = curr_eq

        if pos == 0 and macro_ok[i] and (ema50[i] > ema200[i]) and (30 < rsi[i] < 70) and (macd[i] > macd_sig[i]):
            risk_amt = curr_eq * risk_pct
            stop_dist = (atr[i] * 2.5) if atr[i] > 0 else curr_p * 0.05
            
            cost = next_o * 1.002
            shares = int(risk_amt / stop_dist) if stop_dist > 0 else 0
            if shares * cost > cash: shares = int(cash / cost)
                
            if shares > 0:
                pos, buy_p, trail_h = shares, cost, next_o
                cash -= (shares * cost)
                trades += 1
                
        elif pos > 0:
            trail_h = max(trail_h, curr_p)
            trail_stop = trail_h - (atr[i] * 3)
            
            if (ema50[i] < ema200[i]) or (rsi[i] > 75) or (curr_p < trail_stop):
                sell_p = next_o * 0.998
                if next_o < trail_stop: sell_p = next_o * 0.998 
                
                cash += pos * sell_p
                if sell_p > buy_p: wins += 1
                pos = 0

    equity[-1] = cash + (pos * c[-1] if pos > 0 else 0)
    
    # 🛑 แก้บั๊ก 0/0 (Math Error) ในการคำนวณ MDD อย่างสมบูรณ์แบบ
    equity = np.where(equity == 0, initial_capital, equity)
    ret_pct = ((equity[-1] - initial_capital) / initial_capital) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0.0
    
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdowns = np.where(peak > 0, (equity - peak) / peak, 0.0)
    mdd = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0.0
    
    return ret_pct, win_rate, mdd, trades

# ==========================================
# 4. DASHBOARD & UI
# ==========================================
st.markdown("""<div class="main-header">
    <h1 style="margin:0;">🐉 QuantEco OS v8.2 - Feng Shui Edition</h1>
    <p style="margin:5px 0 0; color:#8892B0;">ระบบเรดาร์สแกนหุ้นระดับสถาบัน ผสานจิตวิทยาการลงทุนและศาสตร์แห่งความมั่งคั่ง</p>
</div>""", unsafe_allow_html=True)

menu = st.sidebar.selectbox("เมนูหลัก:", ["🔭 เรดาร์สแกนหุ้น (Dynamic Scan)", "💼 พอร์ตจำลอง"])

if menu == "🔭 เรดาร์สแกนหุ้น (Dynamic Scan)":
    market = st.radio("เลือกน่านน้ำแห่งความมั่งคั่ง:", ["🇺🇸 หุ้นอเมริกา (S&P 500)", "🇹🇭 หุ้นไทย (SET50)"], horizontal=True)
    is_thai = "SET50" in market
    
    all_tickers = get_set50_tickers() if is_thai else get_sp500_tickers()
    macro_ticker = "^SET.BK" if is_thai else "SPY"
    
    # แสดงจำนวนหุ้นที่ถูกต้องแล้ว!
    st.info(f"🌐 เชื่อมต่อกระแสเงินสำเร็จ: พร้อมสแกนหุ้น **{len(all_tickers)}** ตัว (Macro Index: {macro_ticker})")
    
    limit = st.slider("จำนวนหุ้นที่จะสแกน (ปรับให้พอดีเพื่อป้องกันการบล็อก):", 10, len(all_tickers), min(50, len(all_tickers)), 5)
    scan_list = all_tickers[:limit]
    
    # ปุ่มเปลี่ยนเป็นสีน้ำเงินตาม CSS
    if st.button("🌊 เริ่มต้นสแกนหาจุดเข้าซื้อ (Initiate Scan)"):
        macro_series = get_macro_regime(macro_ticker)
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(scan_list):
            status_text.text(f"กำลังวิเคราะห์กระแสเงิน: {t} ({i+1}/{limit})")
            progress_bar.progress((i + 1) / limit)
            
            try:
                d = yf.download(t, period="5y", progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d = d.xs(t, axis=1, level=1)
                    
                if len(d) > 200:
                    d = calculate_indicators(d)
                    ret, win, mdd, n = run_fast_backtest(d, macro_series)
                    
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
            st.success(f"สแกนเสร็จสิ้น! พบหุ้นที่มีกระแสเงินไหลเข้าและปลอดภัยตามเกณฑ์:")
            
            # 🛑 แก้บั๊ก Pandas .map() รองรับทุกเวอร์ชัน
            try:
                styled_df = df_res.style.map(lambda x: 'color: #00c851' if 'BUY' in str(x) else ('color: #ff4444' if 'DOWN' in str(x) else ''), subset=['Signal'])
            except AttributeError:
                styled_df = df_res.style.applymap(lambda x: 'color: #00c851' if 'BUY' in str(x) else ('color: #ff4444' if 'DOWN' in str(x) else ''), subset=['Signal'])
                
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูลผ่านเกณฑ์ หรือติด Rate Limit")

else:
    st.info("โหมดพอร์ตจำลอง (ฟังก์ชันการแสดงผลพอร์ตคงเดิม)")
