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

# デフォルトの名言リスト
INVESTMENT_QUOTES = [
    "「ルール1：絶対にお金を損しないこと」— バフェット",
    "「あなたがパニックで売る時、誰かが笑って買っている」",
    "「強気相場は、悲観の中に生まれ、懐疑の中に育つ」",
    "「投資で一番大切なのは、頭脳ではなく忍耐強さだ」",
    "「卵を一つのカゴに盛るな」— 投資の格言",
    "「市場が強欲な時に恐れ、恐れている時に強欲になれ」"
]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. セッション状態の管理 ---
# キャラクターのつぶやきを保持
if "char_msg" not in st.session_state:
    st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)

# --- 3. CSS：透過・広告横並び・アニメーション ---
st.markdown(f"""
    <style>
    /* メイン要素 */
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
    div[data-testid="stMetric"] {{ background-color: rgba(150, 150, 150, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.3); }}
    
    /* 広告コンテナ（横並び） */
    .ad-container {{ 
        display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 25px 0; 
    }}
    .ad-card {{ 
        flex: 1; min-width: 250px; max-width: 400px; padding: 20px; 
        border: 2px dashed rgba(150, 150, 150, 0.5); border-radius: 15px; 
        text-align: center; background-color: rgba(150, 150, 150, 0.05); 
    }}
    .ad-card a {{ text-decoration: none; color: #3182ce; font-weight: bold; font-size: 1.1em; }}

    /* キャラクター固定配置（最前面） */
    .floating-char-box {{
        position: fixed; bottom: 20px; right: 20px; z-index: 999;
        display: flex; flex-direction: column; align-items: center; pointer-events: none;
    }}
    
    /* キャラクター画像：白いフチ対策のフィルタ */
    .char-img {{
        width: 140px; height: auto;
        mix-blend-mode: multiply;
        filter: contrast(125%) brightness(108%) drop-shadow(5px 5px 15px rgba(0,0,0,0.3));
        animation: float 3s ease-in-out infinite;
    }}

    /* 吹き出しデザイン */
    .auto-quote-bubble {{
        background: white; border: 2px solid #3182ce; border-radius: 15px;
        padding: 10px 15px; margin-bottom: 10px; font-size: 0.85em; font-weight: bold; color: #1a202c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15); width: 220px; text-align: center; position: relative;
    }}
    .auto-quote-bubble::after {{
        content: ""; position: absolute; bottom: -10px; right: 45%;
        border-width: 10px 10px 0; border-style: solid; border-color: #ffffff transparent;
    }}

    /* 透明ポップオーバー（クリック判定） */
    div[data-testid="stPopover"] {{ position: fixed; bottom: 20px; right: 20px; z-index: 1000; }}
    div[data-testid="stPopover"] > button {{
        width: 140px !important; height: 200px !important;
        background: transparent !important; color: transparent !important; border: none !important;
        box-shadow: none !important; cursor: pointer;
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-12px); }}
    }}

    .advice-box {{ padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. キャラクターとつぶやき表示 ---
st.markdown(f"""
    <div class="floating-char-box">
        <div class="auto-quote-bubble">{st.session_state.char_msg}</div>
        <img src="{CHARACTER_URL}" class="char-img">
    </div>
    """, unsafe_allow_html=True)

# キャラクリック時の挙動（手動で名言更新）
with st.popover(""):
    st.markdown("### 📜 アイモンの知恵")
    st.write(st.session_state.char_msg)
    if st.button("名言をシャッフル"):
        st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
        st.rerun()

# --- 5. メイン画面：市場指標 ---
st.title("🤖 AIマーケット総合診断 Pro")
st.caption("最新AIが市場を予測。診断結果に合わせて右下のキャラがつぶやきます！")

@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"ドル円": "JPY=X", "日経平均": "^N225", "NYダウ": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="1mo", progress=False)
            if not info.empty:
                curr = info['Close'].iloc[-1]
                prev = info['Close'].iloc[-2]
                data[name] = (float(curr), float(curr - prev))
        except: data[name] = (None, None)
    return data

idx_data = get_market_indices()
m1, m2, m3 = st.columns(3)
if idx_data.get('ドル円') and idx_data['ドル円'][0]:
    m1.metric("💴 ドル/円", f"{idx_data['ドル円'][0]:,.2f}円", f"{idx_data['ドル円'][1]:+,.2f}")
if idx_data.get('日経平均') and idx_data['日経平均'][0]:
    m2.metric("🇯🇵 日経平均", f"{idx_data['日経平均'][0]:,.2f}円", f"{idx_data['日経平均'][1]:+,.2f}")
if idx_data.get('NYダウ') and idx_data['NYダウ'][0]:
    m3.metric("🇺🇸 NYダウ", f"{idx_data['NYダウ'][0]:,.2f}ドル", f"{idx_data['NYダウ'][1]:+,.2f}")

st.markdown("---")

# --- 6. 銘柄入力・条件設定 ---
st.markdown("<div class='main-step'>STEP 1: 銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T"}
c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("リストから選択", list(stock_presets.keys()), default=["エヌビディア"])
free_input = c_in2.text_input("直接入力 (例: MSFT, 9984.T)", "")

final_targets = {name: stock_presets[name] for name in selected_names}
if free_input:
    clean_input = free_input.strip().upper()
    final_targets[clean_input] = clean_input

st.markdown("<div class='main-step'>STEP 2: 条件設定</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
f_inv = c1.number_input("投資シミュレーション金額(円)", min_value=1000, value=100000)
time_span = c2.select_slider("分析期間", options=["1週間", "30日", "1年", "5年"], value="30日")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y"}

execute = st.button("🚀 AI診断スタート！")

# 広告（横並び配置）
st.markdown("""<div class="ad-container">
    <div class="ad-card">
        <p>📊 証券口座なら</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">DMM 株 口座開設 [PR]</a>
    </div>
    <div class="ad-card">
        <p>📱 投資アプリなら</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">投資アプリ TOSSY [PR]</a>
    </div>
</div>""", unsafe_allow_html=True)

# --- 7. 診断ロジック & キャラ連動 ---
if "sentiment_analyzer" not in st.session_state:
    st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

if execute and final_targets:
    results, plot_data = [], {}
    sentiments = []
    
    with st.spinner('AIが市場データを解析中...'):
        for name, symbol in final_targets.items():
            try:
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data[name] = df
                
                # 簡易予測と感情分析
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                pred = float(LinearRegression().fit(X_reg, y_reg).predict([[len(y_reg)+5]])[0][0])
                
                # ニュース取得
                q = name if ".T" in symbol else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ja&gl=JP"
                feed = feedparser.parse(url)
                score = 3
                if feed.entries:
                    s_list = [int(st.session_state.sentiment_analyzer(e.title[:128])[0]['label'].split()[0]) for e in feed.entries[:2]]
                    score = sum(s_list)/len(s_list)
                
                sentiments.append(score)
                adv, col = ("🌟強気判定", "#d4edda") if score >= 3.5 and pred > curr else ("⚠️警戒判定", "#f8d7da") if score <= 2.2 else ("😐様子見", "#e2e3e5")
                results.append({"銘柄": name, "将来": f_inv * (pred / curr), "adv": adv, "col": col})
            except: continue

    # ★キャラのセリフ更新ロジック★
    if sentiments:
        avg_s = sum(sentiments) / len(sentiments)
        if avg_s >= 3.7:
            st.session_state.char_msg = "全体的にかなりポジティブだね！この波に乗っちゃう？🚀"
        elif avg_s <= 2.3:
            st.session_state.char_msg = "ちょっと怖い雰囲気を感じるよ…慎重にね！☔"
        else:
            st.session_state.char_msg = "分析完了！今は落ち着いた動きが続きそうだね。☕"
    
    # 結果表示
    if results:
        st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        japanize_matplotlib.japanize()
        for name, data in plot_data.items():
            ax.plot(data.index, data['Close']/data['Close'].iloc[0]*100, label=name)
        ax.legend(); st.pyplot(fig)
        
        for res in results:
            c_res1, c_res2 = st.columns([1, 2])
            c_res1.metric(res['銘柄'], f"{res['将来']:,.0f}円", f"{res['将来']-f_inv:+,.0f}円")
            c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
        
        st.rerun() # セリフを即時反映

# --- 8. 免責事項 ---
st.markdown("""<div style="font-size: 0.8em; opacity: 0.6; padding: 20px; border-top: 1px solid #eee; margin-top: 50px;">
    ⚠️ 免責事項: 投資は自己責任でお願いします。本アプリの予測は将来の成果を保証するものではありません。[PR]広告が含まれています。
</div>""", unsafe_allow_html=True)
