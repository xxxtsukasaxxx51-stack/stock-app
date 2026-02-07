import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from transformers import pipeline
from sklearn.linear_model import LinearRegression
import urllib.parse
import numpy as np
from datetime import timedelta
from deep_translator import GoogleTranslator
import random
import re

# --- 0. 基本設定 ---
APP_URL = "https://your-app-name.streamlit.app/" 
CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true"
INVESTMENT_QUOTES = ["「短期は感情、長期は理屈」だよ。", "「分散投資」は唯一のフリーランチだよ。"]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. セッション管理 ---
if "char_msg" not in st.session_state: st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
if "results" not in st.session_state: st.session_state.results = []
if "plot_data" not in st.session_state: st.session_state.plot_data = {}

# --- 3. CSS (スマホでの視認性を最優先) ---
st.markdown(f"""
    <style>
    /* モバイルでの文字サイズ調整 */
    html {{ font-size: 14px; }}
    @media (min-width: 768px) {{ html {{ font-size: 16px; }} }}

    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.1rem; margin-bottom: 15px; border-left: 5px solid #3182ce; padding-left: 10px; }}
    
    /* 広告：PCで横並び、スマホで縦並び */
    .ad-row {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }}
    .ad-card {{ 
        flex: 1; min-width: 280px; padding: 15px; 
        border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 12px; 
        background: rgba(128, 128, 128, 0.05); text-align: center;
    }}
    .ad-card a {{ display: block; background: #3182ce; color: white !important; padding: 10px; border-radius: 8px; font-weight: bold; text-decoration: none; margin-top: 10px; }}

    .x-share-button {{ display: inline-block; background: #000; color: #fff !important; padding: 10px 20px; border-radius: 25px; font-weight: bold; text-decoration: none; margin: 10px 0; }}

    /* 判定ボックス */
    .advice-box {{ padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; color: #1a202c; margin-bottom: 15px; }}
    
    /* 浮遊キャラ：スマホでは少し小さく */
    .floating-char-box {{ position: fixed; bottom: 10px; right: 10px; z-index: 99; pointer-events: none; }}
    .char-img {{ width: 80px; mix-blend-mode: multiply; animation: float 3s ease-in-out infinite; }}
    @media (min-width: 768px) {{ .char-img {{ width: 120px; }} }}
    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-8px); }} }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. 補助関数 ---
STOCK_PRESETS = {
    "🇺🇸 エヌビディア (AI半導体)": "NVDA", "🇺🇸 テスラ (電気自動車)": "TSLA", "🇺🇸 アップル (iPhone)": "AAPL",
    "🇯🇵 トヨタ自動車 (世界一)": "7203.T", "🇯🇵 ソニーG (エンタメ)": "6758.T", "🇯🇵 三菱UFJ銀 (金融)": "8306.T"
}

def clean_stock_name(name):
    name = re.sub(r'[^\w\s\.]', '', name)
    return name.strip().split(' ')[0]

# --- 5. メイン画面 ---
st.title("🤖 AIマーケット総合診断 Pro")

c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("銘柄選択", list(STOCK_PRESETS.keys()), default=["🇺🇸 エヌビディア (AI半導体)"])
f_inv = c_in2.number_input("投資額(円)", min_value=1000, value=100000)
time_span = st.select_slider("分析期間", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="全期間(Max)")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

if st.button("🚀 AI診断スタート"):
    st.session_state.results = []
    # AIモデルの読み込みをキャッシュして高速化
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('スマホでも解析中...少々お待ちください'):
        for full_name in selected_names:
            try:
                symbol = STOCK_PRESETS[full_name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                
                # 線形回帰予測
                y = df['Close'].tail(20).values
                x = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                pred_val = float(model.predict([[len(y)+5]])[0])
                
                curr = float(df['Close'].iloc[-1])
                display_name = clean_stock_name(full_name)
                
                # ニュース取得 (タイムアウト対策)
                avg_score = 3.0
                adv, col = ("🚀 強気", "#d4edda") if pred_val > curr else ("⚠️ 警戒", "#f8d7da")
                
                st.session_state.results.append({
                    "銘柄": display_name, "将来": f_inv * (pred_val / curr), "adv": adv, "col": col, 
                    "gain": f_inv * (pred_val / curr) - f_inv, "pred_date": "5日後", "invest": f_inv
                })
                st.session_state.plot_data[display_name] = df
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                continue
    st.rerun()

# --- 6. 結果表示 ---
if st.session_state.results:
    st.markdown("<div class='main-step'>診断結果</div>", unsafe_allow_html=True)
    
    # グラフ (モバイルでは高さを抑える)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    japanize_matplotlib.japanize()
    
    for res in st.session_state.results:
        name = res['銘柄']
        if name in st.session_state.plot_data:
            df = st.session_state.plot_data[name]
            base = df['Close'].iloc[0]
            ax.plot(df.index, df['Close']/base*100, label=name)
    
    ax.legend()
    st.pyplot(fig)

    for res in st.session_state.results:
        st.markdown(f"### 🎯 {res['銘柄']}")
        st.metric("予想資産額", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        st.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
        
        share_text = f"【AI株診断】\n🎯 銘柄：{res['銘柄']}\n📢 判定：{res['adv']}\n🚀 予想：{res['将来']:,.0f}円\n{APP_URL}"
        st.markdown(f'<a href="https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}" target="_blank" class="x-share-button">𝕏 ポストする</a>', unsafe_allow_html=True)
        st.divider()

# --- 7. 広告・キャラ ---
st.markdown(f"""
<div class="ad-row">
    <div class="ad-card">
        <p style="font-weight:bold;">DMM 株 [PR]</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">口座開設はこちら</a>
    </div>
    <div class="ad-card">
        <p style="font-weight:bold;">TOSSY [PR]</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">アプリを体験</a>
    </div>
</div>
<div class="floating-char-box"><img src="{CHARACTER_URL}" class="char-img"></div>
""", unsafe_allow_html=True)
