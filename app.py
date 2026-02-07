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
# 手順通りにGitHubへアップした画像の「raw」URLをここに貼ってください
CHARACTER_URL = "https://raw.githubusercontent.com/あなたのユーザー名/stock-app/main/character.png"

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "AIzaSyC4kqvsdMNVr1tIHFLIDSSZa4oudBtki5g"

genai.configure(api_key=GOOGLE_API_KEY)
model_chat = genai.GenerativeModel('gemini-pro')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide", page_icon="🤖")

# --- 2. CSS：デザイン統合（白抜き対策・巨大キャラ・吹き出し・広告） ---
st.markdown(f"""
    <style>
    /* メイン画面装飾 */
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
    div[data-testid="stMetric"] {{ background-color: rgba(150, 150, 150, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.3); }}
    .news-box {{ padding: 12px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.5); margin-bottom: 10px; }}
    .news-box a {{ text-decoration: none; color: #4dabf7 !important; }}
    .advice-box {{ padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; border: 2px solid rgba(150, 150, 150, 0.3); color: #1a1a1a; }}
    
    /* 広告コンテナ（スマホ最適化） */
    .ad-container {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 20px 0; }}
    .ad-card {{ flex: 1; min-width: 280px; max-width: 500px; padding: 20px; border: 2px dashed rgba(150, 150, 150, 0.5); border-radius: 15px; background-color: rgba(150, 150, 150, 0.05); text-align: center; }}

    /* キャラクターと吹き出しの固定配置 */
    .floating-char-container {{
        position: fixed;
        bottom: 100px;
        right: 20px;
        z-index: 999;
        display: flex;
        flex-direction: column;
        align-items: center;
        pointer-events: none;
    }}
    .char-img {{
        width: 130px; /* 大きめに設定 */
        height: auto;
        mix-blend-mode: multiply; /* 白い背景を透過 */
        filter: drop-shadow(5px 5px 10px rgba(0,0,0,0.3));
        animation: float 3s ease-in-out infinite;
    }}
    .bubble {{
        position: relative; background: #ffffff; border: 2px solid #3182ce; border-radius: 15px;
        padding: 8px 12px; margin-bottom: 10px; font-size: 0.85em; color: #1a1a1a;
        max-width: 180px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); font-weight: bold;
    }}
    .bubble::after {{
        content: ""; position: absolute; bottom: -10px; right: 20px;
        border-width: 10px 10px 0; border-style: solid; border-color: #ffffff transparent;
    }}
    @keyframes float {{
        0% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-15px) rotate(2deg); }}
        100% {{ transform: translateY(0px) rotate(0deg); }}
    }}

    /* チャットボタンの固定 */
    div[data-testid="stPopover"] {{ position: fixed; bottom: 30px; right: 25px; z-index: 1000; }}
    .disclaimer-box {{ font-size: 0.8em; opacity: 0.8; background-color: rgba(150, 150, 150, 0.1); padding: 20px; border-radius: 10px; line-height: 1.6; margin-top: 50px; border: 1px solid rgba(150, 150, 150, 0.2); }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. つぶやきとキャラクター表示 ---
monologue = [
    "今のマーケット、AI的にはどうかな？", "長期投資は『気絶』して待つのも手だよ！",
    "分散投資は基本！卵は分けて盛ろうね。", "ニュースの星が多い時はチャンスかも？",
    "無理な取引はダメだよ。心に余裕を✨", "エヌビディアの勢い、凄いね…！"
]
st.markdown(f"""
    <div class="floating-char-container">
        <div class="bubble">{random.choice(monologue)}</div>
        <img src="{CHARACTER_URL}" class="char-img">
    </div>
    """, unsafe_allow_html=True)

# --- 4. メインヘッダーと指標 ---
st.title("🤖 AIマーケット総合診断 Pro")
st.caption("最新AIが市場を予測。右下のアイモンにいつでも相談してね！")

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
    if d[0]: col.metric(lab, f"{d[0]:,.2f}{u}", f"{d[1]:+,.2f}")
    else: col.metric(lab, "取得中...", "休止")
disp_m(m1, "💴 ドル/円", idx_data['ドル円'], "円")
disp_m(m2, "🇯🇵 日経平均", idx_data['日経平均'], "円")
disp
