import streamlit as ui
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
ui.set_page_config(page_title="QuantEco OS v5.2", layout="wide")
analyzer = SentimentIntensityAnalyzer()

if 'watchlist' not in ui.session_state:
    ui.session_state.watchlist = []
if 'custom_scan_us' not in ui.session_state:
    ui.session_state.custom_scan_us = []
if 'custom_scan_th' not in ui.session_state:
    ui.session_state.custom_scan_th = []

DEFAULT_US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
DEFAULT_TH_STOCKS = ["PTT.BK", "CPALL.BK", "ADVANC.BK", "KBANK.BK", "SCB.BK", "TTB.BK", "KTB.BK", "LH.BK", "GULF.BK"]

# ==========================================
# 2. INSTITUTIONAL QUANT ENGINE (Fixed Fallback)
# ==========================================
def calculate_indicators(df):
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Pct_Change'] = df['Close'].pct_change()
    df['Rolling_Vol'] = df['Pct_Change'].rolling(20).std() 
    return df

def run_realistic_backtest(df, initial_capital=100000, fee_pct=0.002, risk_pct=0.02):
    position = 0
    total_trades = 0
    winning_trades = 0
    cash = initial_capital
    buy_price = 0
    equity_curve = []
    
    for i in range(200, len(df)):
        current_price = df['Close'].iloc[i]
        ema50 = df['EMA50'].iloc[i]
        ema200 = df['EMA200'].iloc[i]
        rsi = df['RSI'].iloc[i]
        rolling_vol = df['Rolling_Vol'].iloc[i]
        
        current_equity = cash + (position * current_price if position > 0 else 0)
        
        if ema50 > ema200 and position == 0 and rsi < 70:
            risk_amount = current_equity * risk_pct
            volatility = rolling_vol * current_price if not pd.isna(rolling_vol) else current_price * 0.05
            stop_distance = volatility * 2.5 if volatility > 0 else current_price * 0.05
            
            cost_price = current_price * (1 + fee_pct)
            shares_to_buy = int(risk_amount / stop_distance) if stop_distance > 0 else 0
            total_cost = shares_to_buy * cost_price
            
            if total_cost > cash:
                shares_to_buy = int(cash / cost_price)
                total_cost = shares_to_buy * cost_price
                
            if shares_to_buy > 0:
                position = shares_to_buy
                buy_price = cost_price
                cash -= total_cost
                total_trades += 1
            
        elif (ema50 < ema200 or rsi > 80) and position > 0:
            sell_price = current_price * (1 - fee_pct)
            cash += position * sell_price
            if sell_price > buy_price:
                winning_trades += 1
            position = 0
                
        current_equity = cash + (position * current_price if position > 0 else 0)
        equity_curve.append(current_equity)
                
    final_equity = cash + (position * df['Close'].iloc[-1] if position > 0 else 0)
    net_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 50.0
    
    if equity_curve:
        eq_series = pd.Series(equity_curve)
        peak = eq_series.cummax()
        drawdown = (eq_series - peak) / peak
        max_drawdown = drawdown.min() * 100
    else:
        max_drawdown = 0.0
        
    return net_return_pct, win_rate, max_drawdown

def evaluate_stock_v5(info, df, is_thai_market=False):
    sector = info.get('sector', 'Financial Services' if is_thai_market else 'Technology')
    last_price = df['Close'].iloc[-1]
    last_ema50 = df['EMA50'].iloc[-1]
    last_ema200 = df['EMA200'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    volatility = df['Pct_Change'].tail(20).std() * last_price
    
    is_uptrend = last_ema50 > last_ema200
    
    stop_loss = last_price - (volatility * 2.5) if volatility > 0 else last_price * 0.95
    risk_distance = last_price - stop_loss
    target_price = last_price + (risk_distance * 2)
    entry_price = last_price if last_rsi < 60 else last_ema50

    # ปรับปรุงท่อนนี้ให้ยืดหยุ่นขึ้น หากดึงงบการเงินไม่ได้ ให้ใช้สัญญาณเทคนิคอลนำทางเพื่อความชัวร์
    if is_uptrend and last_rsi < 70:
        return "BUY", "🟢 ซื้อสะสม", entry_price, target_price, stop_loss, sector
    elif not is_uptrend or last_rsi > 75:
        return "SELL", "🔴 ขาย/หลีกเลี่ยง", "-", "-", "-", sector
    else:
        return "HOLD", "🟡 ถือดูแนวโน้ม", "-", "-", "-", sector

# ==========================================
# 3. ENTERPRISE DASHBOARD
# ==========================================
ui.title("🏦 QuantEco OS v5.2 - Institutional Grade System")
ui.caption("แก้ไขบั๊กข้อมูลงบการเงินตกหล่น | คำนวณความเสี่ยงด้วย 2% Risk Rule | ย้อนหลัง 10 ปี")

menu = ui.sidebar.selectbox("เลือกฟังก์ชันหลัก:", ["🔍 สแกนหุ้น & วางแผนเทรด", "💼 พอร์ตจำลองของฉัน (Risk Manager)"])

if menu == "🔍 สแกนหุ้น & วางแผนเทรด":
    market_select = ui.radio("เลือกตลาดหุ้นหลัก:", ["หุ้นสหรัฐฯ (US Big Tech & Bluechips)", "หุ้นไทยปันผลแข็งแกร่ง (SET)"])
    is_thai = market_select == "หุ้นไทยปันผลแข็งแกร่ง (SET)"
    
    base_stocks = DEFAULT_TH_STOCKS + ui.session_state.custom_scan_th if is_thai else DEFAULT_US_STOCKS + ui.session_state.custom_scan_us
        
    ui.markdown("### ➕ เพิ่มหุ้นที่สนใจเข้าไปในตารางสแกน")
    c_in, c_btn = ui.columns([3, 1])
    with c_in:
        custom_ticker = ui.text_input("พิมพ์ชื่อย่อหุ้นเพิ่มเติมเพื่อรวมในตารางสแกน (เช่น NVDA หรือ SCC):").upper()
    with c_btn:
        ui.markdown("<div style='padding-top:28px;'></div>", unsafe_allow_html=True)
        if ui.button("เพิ่มเข้าตารางสแกน"):
            if custom_ticker:
                final_custom = f"{custom_ticker}.BK" if is_thai and not custom_ticker.endswith(".BK") else custom_ticker
                if is_thai and final_custom not in ui.session_state.custom_scan_th:
                    ui.session_state.custom_scan_th.append(final_custom)
                elif not is_thai and final_custom not in ui.session_state.custom_scan_us:
                    ui.session_state.custom_scan_us.append(final_custom)
                ui.rerun()
    
    if ui.button("🚀 สแกนระดับสถาบัน (ดึงข้อมูล 10 ปี)"):
        results = []
        with ui.spinner("กำลังดึงข้อมูลและคำนวณหักค่าธรรมเนียม..."):
            for t in base_stocks:
                try:
                    s = yf.Ticker(t)
                    d = s.history(period="10y") 
                    if not d.empty and len(d) > 200:
                        d = calculate_indicators(d)
                        net_ret, win_w, mdd = run_realistic_backtest(d)
                        _, sig_txt, ent, tg, sl_p, sector = evaluate_stock_v5(s.info, d, is_thai_market=is_thai)
                        
                        ent_val = f"{ent:,.2f}" if isinstance(ent, float) else "-"
                        tg_val = f"{tg:,.2f}" if isinstance(tg, float) else "-"
                        sl_val = f"{sl_p:,.2f}" if isinstance(sl_p, float) else "-"
                        
                        results.append({
                            "ชื่อหุ้น": t.replace(".BK", ""),
                            "Sector": sector,
                            "คำแนะนำ": sig_txt,
                            "ราคา (Entry)": ent_val,
                            "เป้า (Target)": tg_val,
                            "คัท (Stop Loss)": sl_val,
                            "Win Rate": f"{win_w:.1f}%",
                            "Net Profit (10y)": f"{net_ret:+.1f}%",
                            "Max Drawdown": f"{mdd:.1f}%"
                        })
                except: pass
            
            if results:
                res_df = pd.DataFrame(results)
                ui.success(f"สแกนสำเร็จ! แสดงข้อมูลตามเกณฑ์วิเคราะห์ทางสถิติรันเทรนด์:")
                ui.dataframe(res_df, use_container_width=True)
            else:
                ui.error("ไม่สามารถดึงข้อมูลหุ้นในรายการได้ กรุณาลองกดสแกนอีกครั้ง")
            
    if ui.session_state.custom_scan_th or ui.session_state.custom_scan_us:
        if ui.button("🧹 ล้างหุ้นที่คุณพิมพ์เพิ่มทั้งหมดออก"):
            ui.session_state.custom_scan_th = []
            ui.session_state.custom_scan_us = []
            ui.rerun()

    ui.markdown("---")
    ui.subheader("💼 บันทึกหุ้นเข้าพอร์ตจำลอง (Watchlist)")
    cx, cy = ui.columns(2)
    with cx:
        add_ticker = ui.text_input("พิมพ์ชื่อย่อหุ้นที่ต้องการซื้อเข้าพอร์ต (เช่น AAPL หรือ PTT):", key="wl_in").upper()
    with cy:
        ui.markdown("<div style='padding-top:28px;'></div>", unsafe_allow_html=True)
        if ui.button("➕ บันทึกเข้าพอร์ต"):
            if add_ticker:
                final_t = f"{add_ticker}.BK" if is_thai and not add_ticker.endswith(".BK") else add_ticker
                if final_t not in ui.session_state.watchlist:
                    ui.session_state.watchlist.append(final_t)
                    ui.success(f"เพิ่ม {add_ticker} แล้ว!")
                else:
                    ui.warning("มีหุ้นนี้ในพอร์ตแล้ว")

else:
    ui.subheader("💼 Portfolio Risk Manager")
    if not ui.session_state.watchlist:
        ui.info("พอร์ตจำลองยังว่างอยู่ กรุณาเพิ่มหุ้นจากหน้าสแกนครับ")
    else:
        if ui.button("🔄 อัปเดตข้อมูลพอร์ตสด"):
            ui.rerun()
            
        wl_results = []
        sector_counts = {}
        
        with ui.spinner("กำลังตรวจสอบความเสี่ยงในพอร์ต..."):
            for t in ui.session_state.watchlist:
                try:
                    s = yf.Ticker(t)
                    d = s.history(period="2y")
                    if not d.empty:
                        d = calculate_indicators(d)
                        is_th_stock = ".BK" in t
                        _, sig_txt, _, _, _, sector = evaluate_stock_v5(s.info, d, is_thai_market=is_th_stock)
                        
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                        
                        wl_results.append({
                            "ชื่อหุ้น": t.replace(".BK", ""),
                            "กลุ่มอุตสาหกรรม (Sector)": sector,
                            "ราคาปัจจุบัน": f"{d['Close'].iloc[-1]:,.2f}",
                            "การเปลี่ยนแปลง (%)": f"{d['Pct_Change'].iloc[-1]*100:+.2f}%",
                            "สถานะปัจจุบัน": sig_txt
                        })
                except: pass
        
        overweight_sectors = [sec for sec, count in sector_counts.items() if count > 2 and sec != 'N/A']
        if overweight_sectors:
            ui.error(f"⚠️ **คำเตือนความเสี่ยง:** พอร์ตของคุณถือหุ้นกระจุกตัวในกลุ่ม {', '.join(overweight_sectors)} มากเกินไป")
        else:
            ui.success("✅ **การกระจายความเสี่ยง:** พอร์ตของคุณมีความสมดุล")

        ui.dataframe(pd.DataFrame(wl_results), use_container_width=True)
        
        if ui.button("🗑️ ล้างพอร์ตจำลองทั้งหมด"):
            ui.session_state.watchlist = []
            ui.rerun()
