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

# --- 0. 基本設定 ---
CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true"

INVESTMENT_QUOTES = [
    "「ルール1：絶対にお金を損しないこと」— バフェット",
    "「あなたがパニックで売る時、誰かが笑って買っている」",
    "「強気相場は、悲観の中に生まれ、懐疑の中に育つ」",
    "「投資で一番大切なのは、頭脳ではなく忍耐強さだ」",
    "「卵を一つのカゴに盛るな」— 投資の格言"
]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. セッション状態の管理 ---
if "char_msg" not in st.session_state:
    st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
if "results" not in st.session_state:
    st.session_state.results = None
if "plot_data" not in st.session_state:
    st.session_state.plot_data = None

# --- 3. CSS：透過・広告横並び ---
st.markdown(f"""
    <style>
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
    div[data-testid="stMetric"] {{ background-color: rgba(150, 150, 150, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.3); }}
    
    .ad-container {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 25px 0; }}
    .ad-card {{ flex: 1; min-width: 250px; max-width: 400px; padding: 20px; border: 2px dashed rgba(150, 150, 150, 0.5); border-radius: 15px; text-align: center; background-color: rgba(150, 150, 150, 0.05); }}
    .ad-card a {{ text-decoration: none; color: #3182ce; font-weight: bold; }}

    .floating-char-box {{ position: fixed; bottom: 20px; right: 20px; z-index: 999; display: flex; flex-direction: column; align-items: center; pointer-events: none; }}
    .char-img {{
        width: 140px; mix-blend-mode: multiply;
        filter: contrast(130%) brightness(110%) saturate(110%);
        animation: float 3s ease-in-out infinite;
    }}
    .auto-quote-bubble {{
        background: white; border: 2px solid #3182ce; border-radius: 15px;
        padding: 10px 15px; margin-bottom: 10px; font-size: 0.85em; font-weight: bold; color: #1a202c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15); width: 220px; text-align: center; position: relative;
    }}
    .auto-quote-bubble::after {{ content: ""; position: absolute; bottom: -10px; right: 45%; border-width: 10px 10px 0; border-style: solid; border-color: #ffffff transparent; }}

    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-12px); }} }}
    .advice-box {{ padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. キャラクターとつぶやき ---
st.markdown(f"""
    <div class="floating-char-box">
        <div class="auto-quote-bubble">{st.session_state.char_msg}</div>
        <img src="{CHARACTER_URL}" class="char-img">
    </div>
    """, unsafe_allow_html=True)

# --- 5. メイン画面：市場指標 ---
st.title("🤖 AIマーケット総合診断 Pro")

@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"ドル円": "JPY=X", "日経平均": "^N225", "NYダウ": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="1mo", progress=False)
            if not info.empty:
                curr, prev = info['Close'].iloc[-1], info['Close'].iloc[-2]
                data[name] = (float(curr), float(curr - prev))
        except: data[name] = (None, None)
    return data

idx_data = get_market_indices()
m1, m2, m3 = st.columns(3)
if "ドル円" in idx_data: m1.metric("💴 ドル/円", f"{idx_data['ドル円'][0]:,.2f}円", f"{idx_data['ドル円'][1]:+,.2f}")
if "日経平均" in idx_data: m2.metric("🇯🇵 日経平均", f"{idx_data['日経平均'][0]:,.2f}円", f"{idx_data['日経平均'][1]:+,.2f}")
if "NYダウ" in idx_data: m3.metric("🇺🇸 NYダウ", f"{idx_data['NYダウ'][0]:,.2f}ドル", f"{idx_data['NYダウ'][1]:+,.2f}")

st.markdown("---")

# 銘柄入力
st.markdown("<div class='main-step'>STEP 1: 銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "トヨタ": "7203.T", "ソニー": "6758.T"}
c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("リストから選択", list(stock_presets.keys()), default=["エヌビディア"])
free_input = c_in2.text_input("直接入力 (例: MSFT, 9984.T)", "")

final_targets = {name: stock_presets[name] for name in selected_names}
if free_input:
    clean_input = free_input.strip().upper()
    final_targets[clean_input] = clean_input

st.markdown("<div class='main-step'>STEP 2: 条件設定</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
f_inv = c1.number_input("金額(円)", min_value=1000, value=100000)
time_span = c2.select_slider("分析期間", options=["1週間", "30日", "1年", "5年"], value="30日")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y"}

# --- 6. 診断実行 ---
if st.button("🚀 AI診断スタート！"):
    results_temp, plot_data_temp = [], {}
    sentiments = []
    
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('市場データを解析中...'):
        for name, symbol in final_targets.items():
            try:
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data_temp[name] = df
                
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                pred = float(LinearRegression().fit(X_reg, y_reg).predict([[len(y_reg)+5]])[0][0])
                
                q = name if ".T" in symbol else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ja&gl=JP"
                feed = feedparser.parse(url)
                score = 3
                if feed.entries:
                    s_list = [int(st.session_state.sentiment_analyzer(e.title[:128])[0]['label'].split()[0]) for e in feed.entries[:2]]
                    score = sum(s_list)/len(s_list)
                
                sentiments.append(score)
                adv, col = ("🌟強気判定", "#d4edda") if score >= 3.5 and pred > curr else ("⚠️警戒判定", "#f8d7da") if score <= 2.2 else ("😐様子見", "#e2e3e5")
                results_temp.append({"銘柄": name, "将来": f_inv * (pred / curr), "adv": adv, "col": col, "gain": f_inv * (pred / curr) - f_inv})
            except: continue

    # 結果をセッションに保存
    st.session_state.results = results_temp
    st.session_state.plot_data = plot_data_temp
    
    # セリフ更新
    if sentiments:
        avg_s = sum(sentiments) / len(sentiments)
        if avg_s >= 3.7: st.session_state.char_msg = "全体的にポジティブ！チャンス到来かな？🚀"
        elif avg_s <= 2.3: st.session_state.char_msg = "ちょっと雲行きが怪しいね…慎重にいこう☔"
        else: st.session_state.char_msg = "分析完了！落ち着いた動きが続きそうだよ☕"
    st.rerun()

# --- 7. 結果表示エリア ---
if st.session_state.results:
    st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
    
    # チャート表示
    fig, ax = plt.subplots(figsize=(10, 4))
    japanize_matplotlib.japanize()
    for name, data in st.session_state.plot_data.items():
        ax.plot(data.index, data['Close']/data['Close'].iloc[0]*100, label=name)
    ax.legend(); st.pyplot(fig)
    
    # 結果カード表示
    for res in st.session_state.results:
        c_res1, c_res2 = st.columns([1, 2])
        c_res1.metric(res['銘柄'], f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)

# 広告（横並び）
st.markdown("""<div class="ad-container">
    <div class="ad-card">📊 証券口座なら<br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">DMM 株 口座開設 [PR]</a></div>
    <div class="ad-card">📱 投資アプリなら<br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">投資アプリ TOSSY [PR]</a></div>
</div>""", unsafe_allow_html=True)
