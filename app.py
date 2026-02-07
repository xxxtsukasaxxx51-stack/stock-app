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
import io
import google.generativeai as genai
import random

# --- 0. 基本設定とキャラクター画像URL ---
CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true"

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "AIzaSyC4kqvsdMNVr1tIHFLIDSSZa4oudBtki5g"

genai.configure(api_key=GOOGLE_API_KEY)
model_chat = genai.GenerativeModel('gemini-pro')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide", page_icon="🤖")

# --- 2. CSS：キャラクタークリック起動・透過・デザイン ---
st.markdown(f"""
    <style>
    /* メイン装飾 */
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
    div[data-testid="stMetric"] {{ background-color: rgba(150, 150, 150, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.3); }}
    .news-box {{ padding: 12px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.5); margin-bottom: 10px; }}
    .news-box a {{ text-decoration: none; color: #4dabf7 !important; }}
    .advice-box {{ padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; border: 2px solid rgba(150, 150, 150, 0.3); color: #1a1a1a; }}
    
    /* 広告コンテナ */
    .ad-container {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 20px 0; }}
    .ad-card {{ flex: 1; min-width: 280px; max-width: 500px; padding: 20px; border: 2px dashed rgba(150, 150, 150, 0.5); border-radius: 15px; background-color: rgba(150, 150, 150, 0.05); text-align: center; }}

    /* キャラクターコンテナ（最前面へ） */
    .floating-container {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}

    /* キャラクター画像 */
    .char-img {{
        width: 140px;
        height: auto;
        mix-blend-mode: multiply;
        filter: contrast(110%) brightness(105%) drop-shadow(5px 5px 15px rgba(0,0,0,0.3));
        animation: float 3s ease-in-out infinite;
        pointer-events: none; /* 画像自体はクリックを透過させる */
    }}

    /* 透明ボタンをキャラに被せる */
    div[data-testid="stPopover"] {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 10000;
    }}
    div[data-testid="stPopover"] > button {{
        width: 140px !important;
        height: 140px !important;
        background-color: transparent !important;
        color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    /* 吹き出し */
    .bubble {{
        position: relative; background: white; border: 2px solid #3182ce; border-radius: 15px;
        padding: 8px 12px; margin-bottom: 10px; font-size: 0.8em; color: black;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2); font-weight: bold; width: 160px; text-align: center;
    }}

    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}

    .disclaimer-box {{ font-size: 0.8em; opacity: 0.8; background-color: rgba(150, 150, 150, 0.1); padding: 20px; border-radius: 10px; line-height: 1.6; margin-top: 50px; border: 1px solid rgba(150, 150, 150, 0.2); }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. キャラクター・チャット配置 ---
# 背面：見た目
current_msg = random.choice(["ボクをタップしてね！", "今の株価どう思う？", "投資の相談にのるよ！"])
st.markdown(f"""
    <div class="floating-container">
        <div class="bubble">{current_msg}</div>
        <img src="{CHARACTER_URL}" class="char-img">
    </div>
    """, unsafe_allow_html=True)

# 前面：透明ボタン
with st.popover(""):
    st.markdown("### 🤖 アイモン投資相談室")
    if "messages" not in st.session_state: st.session_state.messages = []
    chat_c = st.container(height=350)
    for msg in st.session_state.messages: 
        chat_c.chat_message(msg["role"]).markdown(msg["content"])
    
    if prompt := st.chat_input("アイモンに質問..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_c.chat_message("user").markdown(prompt)
        with chat_c.chat_message("assistant"):
            try:
                response = model_chat.generate_content(f"あなたは投資アドバイザーの『アイモン』です。友だちのように優しく答えて。質問：{prompt}")
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except: st.error("エラーが発生しました。")
    if st.button("履歴を消去"):
        st.session_state.messages = []
        st.rerun()

# --- 4. メイン画面 ---
st.title("🤖 AIマーケット総合診断 Pro")
st.caption("最新AIが市場を予測。右下のアイモンをタップして相談してね！")

# 指標
@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"ドル円": "JPY=X", "日経平均": "^N225", "NYダウ": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="1mo", progress=False)
            if not info.empty:
                curr = float(info['Close'].iloc[-1]); prev = float(info['Close'].iloc[-2])
                data[name] = (curr, curr - prev)
            else: data[name] = (None, None)
        except: data[name] = (None, None)
    return data

idx_data = get_market_indices()
m1, m2, m3 = st.columns(3)
def disp_m(col, lab, d, u=""):
    if d[0] is not None: col.metric(lab, f"{d[0]:,.2f}{u}", f"{d[1]:+,.2f}")
    else: col.metric(lab, "取得中...", "休止")
disp_m(m1, "💴 ドル/円", idx_data['ドル円'], "円")
disp_m(m2, "🇯🇵 日経平均", idx_data['日経平均'], "円")
disp_m(m3, "🇺🇸 NYダウ", idx_data['NYダウ'], "ドル")

st.markdown("---")

# --- 5. 銘柄選択・フリー入力 ---
st.markdown("<div class='main-step'>STEP 1: 銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {
    "🇺🇸 米国株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL"},
    "🇯🇵 日本株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T"}
}
all_stocks_preset = {}
for items in stock_presets.values(): all_stocks_preset.update(items)

col_input1, col_input2 = st.columns([2, 1])
with col_input1:
    selected_names = st.multiselect("リストから選択", list(all_stocks_preset.keys()), default=["エヌビディア"])
with col_input2:
    free_input = st.text_input("ティッカー直接入力", placeholder="例: 9984.T, MSFT")

# 選択銘柄と直接入力を統合
final_targets = {name: all_stocks_preset[name] for name in selected_names}
if free_input:
    final_targets[free_input.upper()] = free_input.upper()

st.markdown("<div class='main-step'>STEP 2: 条件設定</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: f_inv = st.number_input("投資金額(円)", min_value=1000, value=100000)
with c2: 
    time_span = st.select_slider("分析期間", options=["1週間", "30日", "1年", "5年", "10年"], value="30日")
    span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","10年":"10y"}

execute = st.button("🚀 AI診断スタート！")

# 広告エリア
st.markdown(f"""
<div class="ad-container">
    <div class="ad-card">
        <p style="font-weight: bold;">📊 証券口座なら</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank" rel="nofollow">
            <div style="padding: 15px; background: #4dabf7; color: white; border-radius: 10px; font-weight: bold;">DMM 株 で口座開設</div>
        </a>
    </div>
    <div class="ad-card">
        <p style="font-weight: bold;">📱 投資アプリなら</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank" rel="nofollow">
            <div style="padding: 15px; background: #51cf66; color: white; border-radius: 10px; font-weight: bold;">投資アプリ TOSSY</div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6. 診断ロジック ---
if "sentiment_analyzer" not in st.session_state:
    st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

if execute and final_targets:
    results, plot_data = [], {}
    with st.spinner('市場データを解析中...'):
        for name, symbol in final_targets.items():
            try:
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data[name] = df
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                pred = float(LinearRegression().fit(X_reg, y_reg).predict([[len(y_reg)]])[0][0])
                
                is_j = ".T" in symbol
                q = name if is_j else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url)
                news_list, stars = [], 0
                if feed.entries:
                    for e in feed.entries[:3]:
                        s = int(st.session_state.sentiment_analyzer(e.title)[0]['label'].split()[0])
                        stars += s
                        title = GoogleTranslator(source='en', target='ja').translate(e.title) if not is_j else e.title
                        news_list.append({"title": title, "score": s, "link": e.link})
                    avg = stars / len(news_list)
                else: avg = 3

                up = pred > curr
                if avg >= 3.5 and up: adv, col = f"🌟【強気】", "#d4edda"
                elif avg <= 2.5 and not up: adv, col = f"⚠️【警戒】", "#f8d7da"
                else: adv, col = f"😐【様子見】", "#e2e3e5"
                results.append({"銘柄": name, "将来": f_inv * (pred / curr), "星": avg, "pred": pred, "news": news_list, "adv": adv, "col": col})
            except: continue

    if results:
        st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        if st.get_option("theme.base") == "dark": plt.style.use('dark_background')
        japanize_matplotlib.japanize()
        for name, data in plot_data.items():
            base = data['Close'].iloc[0]
            line = ax.plot(data.index, data['Close']/base*100, label=name, linewidth=2)
            r = next(i for i in results if i['銘柄'] == name)
            ax.scatter(data.index[-1] + timedelta(days=1), (r['pred']/base)*100, color=line[0].get_color(), marker='*', s=200, edgecolors='white', zorder=10)
        ax.legend(); st.pyplot(fig)

        for res in results:
            st.markdown(f"### 🎯 {res['銘柄']}")
            cr1, cr2 = st.columns([1, 2])
            cr1.metric("予想額", f"{res['将来']:,.0f}円", f"{res['将来']-f_inv:+,.0f}円")
            cr2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
            for n in res['news']:
                st.markdown(f"<div class='news-box'>{'⭐' * n['score']} <a href='{n['link']}' target='_blank'><b>🔗 {n['title']}</b></a></div>", unsafe_allow_html=True)

# --- 7. 免責事項 ---
st.markdown("""<div class="disclaimer-box"><b>⚠️ 免責事項</b><br>投資判断は自己責任で行ってください。本アプリには[PR]広告が含まれています。</div>""", unsafe_allow_html=True)
