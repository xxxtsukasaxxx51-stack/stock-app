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
INVESTMENT_QUOTES = ["「短期は感情、長期は理屈」だよ。", "「分散投資」は唯一のフリーランチだよ。", "「木を見て森も見ず」にならないようにね！"]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. セッション管理 ---
if "char_msg" not in st.session_state: st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
if "results" not in st.session_state: st.session_state.results = []
if "plot_data" not in st.session_state: st.session_state.plot_data = {}

# --- 3. CSS (レスポンシブ & ダークモード & 投稿ボタン) ---
st.markdown(f"""
    <style>
    /* 全体フォント調整 */
    html {{ font-size: 14px; }}
    @media (min-width: 768px) {{ html {{ font-size: 16px; }} }}

    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.1rem; margin-bottom: 15px; border-left: 5px solid #3182ce; padding-left: 10px; }}
    
    /* 広告：PC横並び・スマホ縦並び */
    .ad-row {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 30px 0; width: 100%; }}
    .ad-card {{ 
        flex: 1; min-width: 290px; padding: 20px; 
        border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 15px; 
        background: rgba(128, 128, 128, 0.05); text-align: center;
        display: flex; flex-direction: column; justify-content: space-between;
    }}
    .ad-card a {{ display: block; background: #3182ce; color: white !important; padding: 12px; border-radius: 8px; font-weight: bold; text-decoration: none; margin-top: 10px; }}

    /* Xシェアボタン（ブランドカラー固定） */
    .x-share-button {{
        display: inline-block; background-color: #000000; color: #ffffff !important; 
        padding: 12px 24px; border-radius: 30px; text-decoration: none; 
        font-weight: bold; font-size: 0.9rem; margin: 15px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2); transition: 0.3s;
    }}
    .x-share-button:hover {{ transform: scale(1.02); opacity: 0.9; }}

    .advice-box {{ padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; color: #1a202c; margin-bottom: 15px; }}
    .disclaimer-box {{ font-size: 0.8rem; padding: 20px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.2); margin-top: 40px; line-height: 1.6; }}

    /* キャラクタースタイル */
    .floating-char-box {{ position: fixed; bottom: 20px; right: 20px; z-index: 99; pointer-events: none; }}
    .char-img {{ width: 100px; mix-blend-mode: multiply; filter: contrast(110%); animation: float 3s ease-in-out infinite; }}
    @media (min-width: 768px) {{ .char-img {{ width: 140px; }} }}
    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-10px); }} }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. 銘柄・補助関数 ---
STOCK_PRESETS = {
    "🇺🇸 エヌビディア (AI半導体)": "NVDA", "🇺🇸 テスラ (電気自動車)": "TSLA", "🇺🇸 アップル (iPhone)": "AAPL",
    "🇯🇵 トヨタ自動車 (世界一)": "7203.T", "🇯🇵 ソニーG (エンタメ)": "6758.T", "🇯🇵 三菱UFJ銀 (金融)": "8306.T"
}

def clean_stock_name(name):
    name = re.sub(r'[^\w\s\.]', '', name)
    return name.strip().split(' ')[0]

# --- 5. メイン画面 ---
st.title("🤖 AIマーケット総合診断 Pro")

st.markdown("<div class='main-step'>STEP 1 & 2: 診断条件の設定</div>", unsafe_allow_html=True)
c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("銘柄を選択（複数可）", list(STOCK_PRESETS.keys()), default=["🇺🇸 エヌビディア (AI半導体)"])
f_inv = c_in2.number_input("シミュレーション投資額(円)", min_value=1000, value=100000, step=10000)

time_span = st.select_slider("分析期間を選択", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="全期間(Max)")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

if st.button("🚀 AI診断スタート"):
    st.session_state.results = []
    if "sentiment_analyzer" not in st.session_state:
        # スマホでのメモリ節約のため、軽量な感情分析モデルをロード
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('AIが市場を解析中...'):
        for full_name in selected_names:
            try:
                symbol = STOCK_PRESETS[full_name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                
                # 株価予測ロジック
                y = df['Close'].tail(20).values
                x = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                pred_val = float(model.predict([[len(y)+5]])[0])
                
                curr = float(df['Close'].iloc[-1])
                display_name = clean_stock_name(full_name)
                
                # 判定ロジック
                adv, col = ("🚀 強気", "#d4edda") if pred_val > curr else ("⚠️ 警戒", "#f8d7da")
                
                st.session_state.results.append({
                    "銘柄": display_name, 
                    "将来": f_inv * (pred_val / curr), 
                    "adv": adv, 
                    "col": col, 
                    "gain": f_inv * (pred_val / curr) - f_inv, 
                    "pred_date": (df.index[-1] + timedelta(days=5)).strftime('%m/%d'), 
                    "period": time_span,
                    "invest": f_inv
                })
                st.session_state.plot_data[display_name] = df
            except: continue
    st.rerun()

# --- 6. 結果表示 ---
if st.session_state.results:
    st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
    
    # グラフ表示
    fig, ax = plt.subplots(figsize=(10, 4))
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
        c_res1, c_res2 = st.columns([1, 1])
        c_res1.metric(f"{res['pred_date']} 予想資産額", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
        
        # --- 𝕏 投稿セクション (いい感じの構成) ---
        share_text = (
            f"📈 【AIマーケット診断】\n"
            f"━━━━━━━━━━━━━━\n"
            f"🎯 銘柄：{res['銘柄']}\n"
            f"🔍 期間：{res['period']}\n"
            f"💰 投資額：{res['invest']:,.0f}円\n"
            f"📢 判定：{res['adv']}\n"
            f"🚀 5日後の予想：{res['将来']:,.0f}円\n"
            f"━━━━━━━━━━━━━━\n"
            f"AIが最新トレンドを解析しました！\n"
            f"アプリで今すぐ診断 👇\n"
            f"{APP_URL}"
        )
        x_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"
        st.markdown(f'<a href="{x_url}" target="_blank" class="x-share-button">𝕏 この診断結果をポストする</a>', unsafe_allow_html=True)
        st.divider()

# --- 7. 広告 & 免責 & キャラ ---
st.markdown(f"""
<div class="ad-row">
    <div class="ad-card">
        <div>
            <span style="background:#ff4b4b; color:white; padding:2px 8px; border-radius:5px; font-size:0.7rem; font-weight:bold;">PR</span>
            <p style="font-weight:bold; margin:10px 0;">DMM 株</p>
            <p style="font-size:0.85rem; opacity:0.8;">スマホで最短即日取引！1株から買える手軽さが人気。初心者の方におすすめです。</p>
        </div>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">無料で口座開設</a>
    </div>
    <div class="ad-card">
        <div>
            <span style="background:#ff4b4b; color:white; padding:2px 8px; border-radius:5px; font-size:0.7rem; font-weight:bold;">PR</span>
            <p style="font-weight:bold; margin:10px 0;">高機能チャート TOSSY</p>
            <p style="font-size:0.85rem; opacity:0.8;">AI予測と組み合わせて、より精度の高い投資判断を。プロ仕様の分析をスマホで。</p>
        </div>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">アプリをチェック</a>
    </div>
</div>

<div class="disclaimer-box">
    <b>⚠️ ご利用上の注意</b><br>
    本アプリの予測は、過去のデータに基づいたAIシミュレーションであり、将来の運用成果を保証するものではありません。投資には元本割れのリスクがあります。実際の取引の際は、ご自身の責任において最終的な判断を行ってください。
</div>

<div class="floating-char-box">
    <div style="background:white; color:#1a202c; border:2px solid #3182ce; border-radius:12px; padding:8px; font-size:0.8rem; font-weight:bold; width:180px; text-align:center; margin-bottom:10px; pointer-events:auto;">
        {st.session_state.char_msg}
    </div>
    <img src="{CHARACTER_URL}" class="char-img">
</div>
""", unsafe_allow_html=True)
