import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import urllib.parse
import feedparser
import random
import japanize_matplotlib

# --- 0. 基本設定 ---
APP_URL = "https://your-app-name.streamlit.app/" 

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. CSS ---
st.markdown("""
    <style>
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.1rem; border-left: 5px solid #3182ce; padding-left: 10px; margin: 20px 0 10px 0; }
    .advice-box { padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; color: #1a202c; margin-bottom: 15px; border: 1px solid rgba(0,0,0,0.1); }
    .news-card { background: rgba(128, 128, 128, 0.08); padding: 12px; border-radius: 10px; margin-bottom: 8px; border-left: 5px solid #3182ce; font-size: 0.85rem; display: flex; justify-content: space-between; align-items: center; }
    .news-stars { color: #f6ad55; font-weight: bold; margin-right: 10px; }
    .x-share-button { display: inline-block; background: #000; color: #fff !important; padding: 12px 24px; border-radius: 30px; text-decoration: none; font-weight: bold; margin: 15px 0; }
    .ad-section { background: linear-gradient(135deg, #f6f9fc 0%, #eef2f7 100%); padding: 20px; border-radius: 15px; border: 1px dashed #cbd5e0; text-align: center; margin: 20px 0; }
    .ad-link { color: #2b6cb0; font-weight: bold; text-decoration: none; font-size: 1.1rem; }
    .disclaimer-box { font-size: 0.75rem; padding: 20px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.2); margin-top: 40px; color: gray; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AIマーケット総合診断 Pro")

# --- 🎯 銘柄マスター ---
stock_master = {
    "🇺🇸 米国成長株": {"エヌビディア": "NVDA", "テスラ": "TSLA", "アップル": "AAPL", "マイクロソフト": "MSFT"},
    "🇯🇵 日本主力株": {"トヨタ自動車": "7203.T", "三菱UFJ": "8306.T", "任天堂": "7974.T", "ソニーグループ": "6758.T"},
    "📈 指数・ETF": {"S&P 500 (VOO)": "VOO", "ナスダック100 (QQQ)": "QQQ"}
}
code_to_name = {c: n for cat in stock_master.values() for n, c in cat.items()}
flat_options = {f"[{cat}] {n} ({c})": c for cat, s in stock_master.items() for n, c in s.items()}

st.markdown("<div class='main-step'>STEP 1 & 2: 銘柄条件設定</div>", unsafe_allow_html=True)
c_sel, c_free = st.columns([1, 1])
selected_keys = c_sel.multiselect("🔥 人気銘柄", list(flat_options.keys()))
free_input = c_free.text_input("✍️ 自由入力 (NFLX, 7203.T等)", placeholder="カンマ区切り")

final_symbols = [flat_options[k] for k in selected_keys]
if free_input:
    final_symbols.extend([s.strip().upper() for s in free_input.split(",") if s.strip()])
final_symbols = list(dict.fromkeys(final_symbols))

c_in1, c_in2 = st.columns([1, 1])
f_inv = c_in1.number_input("投資金額(円)", min_value=1000, value=100000)
time_span = st.select_slider("分析期間", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="1年")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

# --- 実行 ---
if st.button("🚀 AI診断スタート"):
    if not final_symbols:
        st.error("銘柄を入力してください。")
    else:
        results, plot_data = [], {}
        with st.spinner('AI予測グラフを描画中...'):
            for symbol in final_symbols:
                try:
                    df = yf.download(symbol, period=span_map[time_span], progress=False)
                    if df.empty: continue
                    y = df['Close'].values.flatten()
                    y_last = y[-20:] if len(y) >= 20 else y
                    model = LinearRegression().fit(np.arange(len(y_last)).reshape(-1, 1), y_last)
                    
                    # 5日後の予測比率を計算
                    pred_ratio = float(model.predict(np.array([[len(y_last)+5]]))[0]) / float(y[-1])
                    stars = round(np.clip(3.0 + (pred_ratio - 1)*10, 1.5, 5.0), 1)
                    
                    results.append({
                        "name": code_to_name.get(symbol, symbol), "symbol": symbol, 
                        "future": f_inv * pred_ratio, "gain": (f_inv * pred_ratio) - f_inv,
                        "adv": "🚀 強気" if pred_ratio > 1 else "⚠️ 警戒",
                        "col": "#d4edda" if pred_ratio > 1 else "#f8d7da",
                        "stars": stars, "period": time_span, "invest": f_inv
                    })
                    plot_data[symbol] = {"df": df, "stars": stars, "pred_ratio": pred_ratio}
                except: continue

        if results:
            st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
            
            # --- 📈 グラフ（星マークプロット版） ---
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            for s, info in plot_data.items():
                d = info["df"]
                # 成長率ベースのライン（最初を100とする）
                base_price = d['Close'].iloc[0]
                line_data = (d['Close'] / base_price) * 100
                p = ax.plot(d.index, line_data, label=f"{s} (⭐{info['stars']})", linewidth=2)
                
                # グラフの最後に「星」をプロット
                last_date = d.index[-1]
                last_val = line_data.iloc[-1]
                # 予測方向（少し未来）に星を置く
                ax.scatter(last_date, last_val, marker='*', s=200, color=p[0].get_color(), edgecolors='black', zorder=5)

            ax.set_ylabel("成長率 (%)")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='upper left', fontsize='small')
            st.pyplot(fig)

            # --- 💰 PR広告 ---
            st.markdown("""<div class="ad-section"><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank" class="ad-link">【PR】DMM 株：最短即日で口座開設！手数料も業界最安水準</a></div>""", unsafe_allow_html=True)

            for res in results:
                st.markdown(f"### 🎯 {res['name']} ({res['symbol']})")
                r1, r2 = st.columns(2)
                r1.metric("5日後の予想資産", f"{res['future']:,.0f}円", f"{res['gain']:+,.0f}円")
                r2.markdown(f"<div class='advice-box' style='background-color:{res['col']};'>{res['adv']} (AI期待値: ⭐{res['stars']})</div>", unsafe_allow_html=True)
                
                share_text = (f"📈 【AIマーケット診断 Pro】\n🎯 {res['name']}({res['symbol']})\n🔍 期待値：⭐{res['stars']}\n🚀 予想：{res['future']:,.0f}円\n{APP_URL}")
                st.markdown(f'<a href="https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}" target="_blank" class="x-share-button">𝕏 結果をポストする</a>', unsafe_allow_html=True)
                st.divider()

st.markdown('<div class="disclaimer-box">⚠️ 免責事項: 本アプリは過去データに基づくシミュレーションです。</div>', unsafe_allow_html=True)


