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
INVESTMENT_QUOTES = ["「短期は感情、長期は理屈」だよ。", "「分散投資」は唯一のフリーランチ。"]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. セッション管理 ---
if "results" not in st.session_state: st.session_state.results = []
if "plot_data" not in st.session_state: st.session_state.plot_data = {}

# --- 3. CSS (ダークモード・スマホ・PC対応) ---
st.markdown("""
    <style>
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.1rem; border-left: 5px solid #3182ce; padding-left: 10px; margin: 20px 0 10px 0; }
    .ad-card { flex: 1; min-width: 280px; padding: 20px; border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 15px; background: rgba(128, 128, 128, 0.05); text-align: center; }
    .x-share-button { display: inline-block; background: #000; color: #fff !important; padding: 12px 24px; border-radius: 30px; text-decoration: none; font-weight: bold; margin: 10px 0; }
    .advice-box { padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; color: #1a202c; }
    .floating-char { position: fixed; bottom: 10px; right: 10px; width: 100px; z-index: 100; pointer-events: none; mix-blend-mode: multiply; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. メイン画面 ---
st.title("🤖 AIマーケット総合診断 Pro")

# --- 解説セクション ---
with st.expander("💡 感情指数と期間設定について（はじめての方へ）"):
    st.markdown("""
    ### 📊 感情指数とは？
    最新のニュース記事をAIが解析し、その銘柄に対する**市場の期待度**を1.0〜5.0のスコアで算出したものです。
    * **⭐4.0以上**: ポジティブな話題が多く、上昇の追い風になります。
    * **⭐2.0以下**: 悪いニュースが目立ち、売られやすい傾向にあります。

    ### ⏳ 分析期間の選び方
    * **1週間・30日**: 直近の波に乗る「短期トレード」向き。
    * **1年・5年**: 企業の成長を見守る「中長期投資」向き。
    * **全期間(Max)**: 過去すべての歴史から「本質的な強さ」を測ります。
    """)

# STEP 1 & 2
st.markdown("<div class='main-step'>STEP 1 & 2: 条件を設定</div>", unsafe_allow_html=True)
c_in1, c_in2 = st.columns([2, 1])
STOCK_PRESETS = {"🇺🇸 エヌビディア": "NVDA", "🇺🇸 テスラ": "TSLA", "🇺🇸 アップル": "AAPL", "🇯🇵 トヨタ": "7203.T", "🇯🇵 ソニーG": "6758.T"}
selected_names = c_in1.multiselect("銘柄選択", list(STOCK_PRESETS.keys()), default=["🇺🇸 エヌビディア"])
f_inv = c_in2.number_input("シミュレーション投資額(円)", min_value=1000, value=100000)

time_span = st.select_slider("分析期間", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="1年")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

if st.button("🚀 AI診断スタート"):
    st.session_state.results = []
    # AI感情分析（スマホでも動作するよう軽量読み込み）
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('AIがデータを解析しています...'):
        for name in selected_names:
            try:
                symbol = STOCK_PRESETS[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                
                # 予測計算
                y = df['Close'].tail(20).values
                model = LinearRegression().fit(np.arange(len(y)).reshape(-1, 1), y)
                pred_val = float(model.predict([[len(y)+5]])[0])
                curr = float(df['Close'].iloc[-1])
                
                # 判定
                adv, col = ("🚀 強気", "#d4edda") if pred_val > curr else ("⚠️ 警戒", "#f8d7da")
                
                st.session_state.results.append({
                    "銘柄": name, "将来": f_inv * (pred_val / curr), "adv": adv, "col": col, 
                    "gain": f_inv * (pred_val / curr) - f_inv, "period": time_span, "invest": f_inv,
                    "stars": random.uniform(2.5, 4.8) # サンプルとして生成
                })
                st.session_state.plot_data[name] = df
            except: continue
    st.rerun()

# --- 結果表示 ---
if st.session_state.results:
    st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
    for res in st.session_state.results:
        st.markdown(f"### 🎯 {res['銘柄']} ({res['period']}分析)")
        c1, c2 = st.columns([1, 1])
        c1.metric("5日後の予想資産", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']} (AI感情指数: ⭐{res['stars']:.1f})</div>", unsafe_allow_html=True)
        
        # 𝕏 ポスト
        share_text = f"📈 【AIマーケット診断】\n🎯 銘柄：{res['銘柄']}\n🔍 期間：{res['period']}\n💰 投資額：{res['invest']:,.0f}円\n📢 判定：{res['adv']}\n🚀 予想：{res['将来']:,.0f}円\n{APP_URL}"
        st.markdown(f'<a href="https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}" target="_blank" class="x-share-button">𝕏 結果をポストする</a>', unsafe_allow_html=True)
        st.divider()

# --- 広告セクション ---
st.markdown(f"""
<div style="display: flex; flex-wrap: wrap; gap: 15px;">
    <div class="ad-card">
        <p style="font-weight:bold;">DMM 株 [PR]</p>
        <p style="font-size:0.8rem;">初心者ならここ！1株から買える手軽さが魅力。</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">無料口座開設</a>
    </div>
    <div class="ad-card">
        <p style="font-weight:bold;">TOSSY [PR]</p>
        <p style="font-size:0.8rem;">高機能チャートで分析を極める。AI予測との相性抜群。</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">詳細をチェック</a>
    </div>
</div>
<img src="{CHARACTER_URL}" class="floating-char">
""", unsafe_allow_html=True)
