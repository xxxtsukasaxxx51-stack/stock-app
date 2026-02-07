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

# --- 0. 設定とキャラクター画像URL ---
# 好きなキャラクター画像のURLに差し替えてください（透過PNGがおすすめ）
CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true" # 例としてアンモナイト（アイモン）

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

genai.configure(api_key=GOOGLE_API_KEY)
model_chat = genai.GenerativeModel('gemini-pro')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide", page_icon="🤖")

# --- 2. キャラクターを右下に置くための特殊CSS ---
st.markdown(f"""
    <style>
    /* メインステップの装飾 */
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
    
    /* 指標カード */
    div[data-testid="stMetric"] {{ 
        background-color: rgba(150, 150, 150, 0.1); 
        padding: 15px; border-radius: 15px; 
        border: 1px solid rgba(150, 150, 150, 0.3); 
    }}

    /* 広告カード */
    .ad-container {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 20px 0; }}
    .ad-card {{ 
        flex: 1; min-width: 280px; max-width: 500px; padding: 20px; 
        border: 2px dashed rgba(150, 150, 150, 0.5); border-radius: 15px; 
        background-color: rgba(150, 150, 150, 0.05); text-align: center; 
    }}

    /* ★ キャラクター画像を右下に固定するスタイル ★ */
    .floating-char {{
        position: fixed;
        bottom: 90px;
        right: 25px;
        width: 80px;
        height: 80px;
        z-index: 999;
        pointer-events: none; /* 画像自体はクリックをスルーして下のボタンに当てる */
        animation: float 3s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-15px); }}
        100% {{ transform: translateY(0px); }}
    }}

    /* ポップオーバー（チャットボタン）をキャラの下に配置 */
    div[data-testid="stPopover"] {{
        position: fixed;
        bottom: 30px;
        right: 20px;
        z-index: 1000;
    }}
    
    .disclaimer-box {{ 
        font-size: 0.8em; opacity: 0.8; 
        background-color: rgba(150, 150, 150, 0.1); 
        padding: 20px; border-radius: 10px; line-height: 1.6; margin-top: 50px; 
    }}
    </style>
    
    <img src="{CHARACTER_URL}" class="floating-char">
    """, unsafe_allow_html=True)

# --- 3. メイン画面の表示 (指標など) ---
st.title("🤖 AIマーケット総合診断 Pro")
st.caption("最新AIが市場を予測。困ったら右下のアイモンに相談してね！")

# 指標表示 (省略せず実装)
@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"ドル円": "JPY=X", "日経平均": "^N225", "NYダウ": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="1mo", progress=False)
            if not info.empty:
                current = float(info['Close'].iloc[-1]); prev = float(info['Close'].iloc[-2])
                data[name] = (current, current - prev)
            else: data[name] = (None, None)
        except: data[name] = (None, None)
    return data

indices_data = get_market_indices()
m_col1, m_col2, m_col3 = st.columns(3)
def display_m(col, label, d, u=""):
    if d[0]: col.metric(label, f"{d[0]:,.2f}{u}", f"{d[1]:+,.2f}")
    else: col.metric(label, "取得中...", "休止")
display_m(m_col1, "💴 ドル/円", indices_data['ドル円'], "円")
display_m(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_m(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

st.markdown("---")

# 診断ステップ
st.markdown("<div class='main-step'>STEP 1: 銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {
    "🇺🇸 米国株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL"},
    "🇯🇵 日本株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T"},
    "⚡ その他": {"ビットコイン": "BTC-USD", "金(Gold)": "GC=F"}
}
all_stocks = {}
for items in stock_presets.values(): all_stocks.update(items)
selected_names = st.multiselect("銘柄選択", list(all_stocks.keys()), default=["エヌビディア"])

st.markdown("<div class='main-step'>STEP 2: 条件設定</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: f_inv = st.number_input("シミュレーション金額(円)", min_value=1000, value=100000)
with c2: 
    time_span = st.select_slider("分析期間", options=["1週間", "30日", "1年", "5年", "10年", "最大"], value="30日")
    span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","10年":"10y","最大":"max"}

execute = st.button("🚀 AI診断スタート！")

# 広告
st.markdown(f"""
<div class="ad-container">
    <div class="ad-card">
        <p style="font-weight: bold;">📊 証券口座なら</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank" rel="nofollow" style="text-decoration: none;">
            <div style="padding: 15px; background: #4dabf7; color: white; border-radius: 10px; font-weight: bold;">DMM 株 で口座開設</div>
        </a>
    </div>
    <div class="ad-card">
        <p style="font-weight: bold;">📱 投資アプリなら</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank" rel="nofollow" style="text-decoration: none;">
            <div style="padding: 15px; background: #51cf66; color: white; border-radius: 10px; font-weight: bold;">投資アプリ TOSSY</div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# 診断ロジック (簡略化して記載、実際のロジックを保持してください)
if execute:
    st.info("AI分析を実行中...")
    # ここに以前のニュース取得・グラフ描画ロジックが入ります

# --- 4. 🌟 キャラクター連動・右下ポップオーバーチャット 🌟 ---
with st.popover("💬 アイモンに相談する"):
    st.markdown("### 🤖 アイモン投資相談室")
    st.caption("この銘柄についてどう思う？など何でも聞いてね。")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_c = st.container(height=300)
    for msg in st.session_state.messages:
        chat_c.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("ここに質問を入力..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_c.chat_message("user").markdown(prompt)

        with chat_c.chat_message("assistant"):
            try:
                full_p = f"あなたは親切な投資アドバイザーの『アイモン』です。投資初心者の質問に友だちのように優しく答えて。質問：{prompt}"
                response = model_chat.generate_content(full_p)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except:
                st.error("APIキーを確認してね！")
    
    if st.button("履歴クリア"):
        st.session_state.messages = []
        st.rerun()

# --- 5. 免責事項 ---
st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ 免責事項</b><br>
        本アプリは情報提供を目的としており、投資勧誘を意図したものではありません。投資判断は自己責任でお願いします。アフィリエイト広告を含みます。
    </div>
""", unsafe_allow_html=True)
