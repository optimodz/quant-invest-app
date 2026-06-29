import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- Setup ---
st.set_page_config(page_title="QuantEco Final", layout="wide")
st.title("🐉 QuantEco OS: Final Radar Edition")

# --- Functions ---
def get_tickers(is_thai):
    if is_thai:
        return ["PTT.BK", "CPALL.BK", "DELTA.BK", "AOT.BK", "KBANK.BK", "SCB.BK", "ADVANC.BK", "GULF.BK", "PTTEP.BK", "SCC.BK"]
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "INTC"]

def calculate_indicators(df):
    close = df['Close']
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    df['EMA200'] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    return df

# --- Main UI ---
market = st.radio("เลือกตลาด:", ["หุ้นสหรัฐฯ", "หุ้นไทย"], horizontal=True)
is_thai = "ไทย" in market
scan_list = get_tickers(is_thai)

if st.button("🚀 สแกนหุ้นแบบเปิดเผยทุกตัว"):
    results = []
    with st.spinner("กำลังวิเคราะห์..."):
        for t in scan_list:
            try:
                df = calculate_indicators(yf.download(t, period="2y", progress=False))
                last = df.iloc[-1]
                
                # Logic: เช็กเงื่อนไขแบบละเอียด
                is_uptrend = float(last['EMA50']) > float(last['EMA200'])
                is_rsi_ok = float(last['RSI']) < 75
                
                if is_uptrend and is_rsi_ok:
                    signal = "🟢 STRONG BUY"
                elif not is_uptrend:
                    signal = "⚪ HOLD: ต่ำกว่าเส้น EMA200"
                else:
                    signal = "⚪ HOLD: RSI สูงเกินไป"
                
                results.append({
                    "Ticker": t,
                    "Signal": signal,
                    "Price": f"{float(last['Close']):,.2f}",
                    "RSI": f"{float(last['RSI']):.1f}"
                })
            except: continue
            
    if results:
        df_res = pd.DataFrame(results)
        # แสดงผลทุกตัวที่สแกน
        st.table(df_res)
    else:
        st.error("ดึงข้อมูลไม่ได้ กรุณาเช็กอินเทอร์เน็ต")
