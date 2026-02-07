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

# --- 0. グラフ表示の安定化設定 ---
import matplotlib
matplotlib.use('Agg')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide", page_icon="🤖")

# カスタムCSS（ダークモード対応版）
st.markdown("""
    <style>
    /* ステップタイトルの色（青系はどちらでも見やすい） */
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }
    
    /* 指標カードの背景（少し透過させて背景色を活かす） */
    div[data-testid="stMetric"] {
        background-color: rgba(150, 150, 150, 0.1);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(150, 150, 150, 0.3);
    }
    
    /* ニュースボックス（ダークモード時は枠を明るく） */
    .news-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(150, 150, 150, 0.5);
        margin-bottom: 10px;
    }
    .news-box a {
        text-decoration: none;
        color: #4dabf7 !important; /* リンクを明るい青に固定 */
    }
    
    /* アドバイスボックス（文字を常に読みやすく） */
    .advice-box {
        padding: 20px;
        border-radius: 15px;
        margin-top: 10px;
        font-size: 1.1em;
        text-align: center;
        border: 2px solid rgba(150, 150, 150, 0.3);
        color: #1a1a1a; /* ここは背景色が明るいので文字は濃い色で固定 */
    }
    
    /* 広告カード */
    .ad-card {
        padding: 15px;
        border: 1px solid rgba(150, 150, 150, 0.3);
        border-radius: 10px;
        background-color: rgba(150, 150, 150, 0.05);
        text-align: center;
    }
    .ad-card p {
        color: inherit !important;
    }
    
    /* 期間ヒント */
    .span-hint {
        background-color: rgba(49, 130, 206, 0.1);
        padding: 12px;
        border-radius: 10px;
        font-size: 0.9em;
        border-left: 5px solid #3182ce;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AIモデルの準備 ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
analyzer = load_ai()

# --- 3. 指標データの取得 ---
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

# --- 4. メイン画面：ヘッダー ---
st.title("🤖 AIマーケット総合診断 Pro")
st.caption("最新AIがニュースと価格トレンドから、明日の市場を予測します。")

m_col1, m_col2, m_col3 = st.columns(3)
def display_metric(col, label, data_tuple, unit=""):
    val, diff = data_tuple
    if val is not None: col.metric(label, f"{val:,.2f}{unit}", f"{diff:+,.2f}")
    else: col.metric(label, "取得中...", "市場休止中")

display_metric(m_col1, "💴 ドル/円", indices_data['ドル円'], "円")
display_metric(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_metric(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

st.markdown("---")

# --- 5. 操作ステップ案内 ---
st.markdown("<div class='main-step'>STEP 1: 診断したい銘柄を選ぼう</div>", unsafe_allow_html=True)

stock_presets = {
    "🇺🇸 米国株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "パランティア": "PLTR"},
    "🇯🇵 日本株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T"},
    "⚡ その他": {"ビットコイン": "BTC-USD", "金
