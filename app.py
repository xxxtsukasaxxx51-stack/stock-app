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

# --- 0. AIチャットの設定 ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

genai.configure(api_key=GOOGLE_API_KEY)
model_chat = genai.GenerativeModel('gemini-pro')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide", page_icon="🤖")

# カスタムCSS（右下チャットボタン & ポップアップ調整）
st.markdown("""
    <style>
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }
    div[data-testid="stMetric"] { background-color: rgba(150, 150, 150, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.3); }
    .news-box { padding: 12px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.5); margin-bottom: 10px; }
    .news-box a { text-decoration: none; color: #4dabf7 !important; }
    .advice-box { padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; border: 2px solid rgba(150, 150, 150, 0.3); color: #1a1a1a; }
    
    .ad-container { display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 20px 0; }
    .ad-card { flex: 1; min-width: 280px; max-width: 500px; padding: 20px; border: 2px dashed rgba(150, 150, 150, 0.5); border-radius: 15px; background-color: rgba(150, 150, 150, 0.05); text-align: center; }
    
    .disclaimer-box { font-size: 0.8em; opacity: 0.8; background-color: rgba(150, 150, 150, 0.1); padding: 20px; border-radius: 10px; line-height: 1.6; margin-top: 50px; }

    /* --- 右下固定チャットセクション --- */
    /* モバイル・PC共通の調整 */
    .stChatFloating {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. メイン画面：ヘッダー ---
st.title("🤖 AIマーケット総合診断 Pro")
st.caption("最新AIがニュースと価格トレンドから、市場を予測。右下のアイモンにいつでも相談してね！")

# 指標データ取得（中略：ロジックは保持）
@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"ドル円": "JPY=X", "日経平均": "^N225", "NYダウ": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="1mo", progress=False)
            if not info.empty:
                current = float(info['Close'].iloc[-1])
                prev = float(info['Close'].iloc[-2])
                data[name] = (current, current - prev)
            else: data[name] = (None, None)
        except: data[name] = (None, None)
    return data

indices_data = get_market_indices()
m_col1, m_col2, m_col3 = st.columns(3)
def display_metric(col, label, data_tuple, unit=""):
    val, diff = data_tuple
    if val is not None: col.metric(label, f"{val:,.2f}{unit}", f"{diff:+,.2f}")
    else: col.metric(label, "取得中...", "市場休止中")
display_metric(m_col1, "💴 ドル/円", indices_data['ドル円'], "円")
display_metric(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_metric(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

st.markdown("---")

# --- 3. 診断ステップ (STEP 1 & 2) ---
st.markdown("<div class='main-step'>STEP 1: 診断したい銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {
    "🇺🇸 米国株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "パランティア": "PLTR"},
    "🇯🇵 日本株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T"},
    "⚡ その他": {"ビットコイン": "BTC-USD", "金(Gold)": "GC=F"}
}
all_stocks = {}
for cat, items in stock_presets.items(): all_stocks.update(items)
selected_names = st.multiselect("気になる銘柄を選択", list(all_stocks.keys()), default=["エヌビディア"])

st.markdown("<div class='main-step'>STEP 2: 条件を決めよう</div>", unsafe_allow_html=True)
set1, set2 = st.columns(2)
with set1: future_investment = st.number_input("シミュレーション金額(円)", min_value=1000, value=100000)
with set2: 
    time_span = st.select_slider("分析する期間", options=["1週間", "30日", "1年", "5年", "10年", "最大期間"], value="30日")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y", "10年": "10y", "最大期間": "max"}

execute = st.button("🚀 AI診断スタート！")

# 広告エリア
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

# 診断ロジック実行（中略：結果表示ロジックは前回同様）
if execute:
    st.info("分析結果はここに表示されます（前回のコードと同様）")
    # ※ここに診断実行プログラムが入ります

# --- 4. 🌟 右下キャラクターチャット 🌟 ---
# ポップオーバー機能を使って「吹き出し」のようなチャットを作ります
with st.container():
    # 右下に固定されるボタンのように見えるポップオーバー
    with st.popover("💬 アイモンに聞く", use_container_width=False):
        st.markdown("### 🤖 アイモン投資相談室")
        st.caption("経済や投資の疑問を何でも聞いてね！")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # チャット履歴表示
        chat_container = st.container(height=300)
        for msg in st.session_state.messages:
            chat_container.chat_message(msg["role"]).markdown(msg["content"])

        if prompt := st.chat_input("例：円安のメリットは？"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            chat_container.chat_message("user").markdown(prompt)

            with chat_container.chat_message("assistant"):
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
    <p style='text-align: center; opacity: 0.5; font-size: 0.7em; margin-top:10px;'>© 2026 AI Market Diagnosis Pro</p>
""", unsafe_allow_html=True)
