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
    "「卵を一つのカゴに盛るな」— 投資の格言",
    "「市場が強欲な時に恐れ、恐れている時に強欲になれ」",
    "「投資で一番大切なのは、頭脳ではなく忍耐強さだ」"
]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. セッション管理 ---
if "char_msg" not in st.session_state:
    st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
if "results" not in st.session_state:
    st.session_state.results = None
if "plot_data" not in st.session_state:
    st.session_state.plot_data = None

# --- 3. CSS：透過・横並び広告・吹き出し ---
st.markdown(f"""
    <style>
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
    div[data-testid="stMetric"] {{ background-color: rgba(150, 150, 150, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.3); }}
    
    .ad-container {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 25px 0; }}
    .ad-card {{ flex: 1; min-width: 250px; max-width: 400px; padding: 20px; border: 2px dashed rgba(150, 150, 150, 0.5); border-radius: 15px; text-align: center; background-color: rgba(150, 150, 150, 0.05); }}
    .ad-card a {{ text-decoration: none; color: #3182ce; font-weight: bold; }}

    .floating-char-box {{ position: fixed; bottom: 20px; right: 20px; z-index: 999; display: flex; flex-direction: column; align-items: center; pointer-events: none; }}
    .char-img {{ width: 140px; mix-blend-mode: multiply; filter: contrast(130%) brightness(110%); animation: float 3s ease-in-out infinite; }}
    .auto-quote-bubble {{
        background: white; border: 2px solid #3182ce; border-radius: 15px;
        padding: 10px 15px; margin-bottom: 10px; font-size: 0.85em; font-weight: bold; color: #1a202c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15); width: 220px; text-align: center; position: relative;
    }}
    .auto-quote-bubble::after {{ content: ""; position: absolute; bottom: -10px; right: 45%; border-width: 10px 10px 0; border-style: solid; border-color: #ffffff transparent; }}

    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-12px); }} }}
    
    .news-box {{ background: white; padding: 12px; border-radius: 8px; border-left: 5px solid #3182ce; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .advice-box {{ padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); }}
    .sentiment-badge {{ background: #edf2f7; padding: 5px 12px; border-radius: 20px; font-weight: bold; color: #2d3748; display: inline-block; margin-bottom: 10px; }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. キャラクター表示 ---
st.markdown(f"""<div class="floating-char-box"><div class="auto-quote-bubble">{st.session_state.char_msg}</div><img src="{CHARACTER_URL}" class="char-img"></div>""", unsafe_allow_html=True)

# --- 5. メイン画面：市場指標 ---
st.title("🤖 AIマーケット総合診断 Pro")
idx_data = get_market_indices() # キャッシュ関数は前述のものを想定
m1, m2, m3 = st.columns(3)
# (指標表示は省略：前述のコードと同様)

st.markdown("---")

# 銘柄入力
st.markdown("<div class='main-step'>STEP 1: 銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "トヨタ": "7203.T", "ソニー": "6758.T"}
c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("リストから選択", list(stock_presets.keys()), default=["エヌビディア"])
free_input = c_in2.text_input("直接入力 (例: MSFT, 9984.T)", "")
final_targets = {name: stock_presets[name] for name in selected_names}
if free_input: final_targets[free_input.upper()] = free_input.upper()

st.markdown("<div class='main-step'>STEP 2: 条件設定</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
f_inv = c1.number_input("金額(円)", min_value=1000, value=100000)
time_span = c2.select_slider("分析期間", options=["1週間", "30日", "1年", "5年"], value="30日")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y"}

# --- 6. 診断実行ロジック ---
if st.button("🚀 AI診断スタート！"):
    results_temp, plot_data_temp = [], {}
    sentiments_all = []
    
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('AIが予測とニュースを分析中...'):
        for name, symbol in final_targets.items():
            try:
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data_temp[name] = df
                
                # 線形回帰予測
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model = LinearRegression().fit(X_reg, y_reg)
                pred = float(model.predict([[len(y_reg)+5]])[0][0])
                
                # ニュースと感情分析
                is_j = ".T" in symbol
                q = name if is_j else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ja&gl=JP"
                feed = feedparser.parse(url)
                news_list, stars_sum = [], 0
                if feed.entries:
                    for e in feed.entries[:3]:
                        s = int(st.session_state.sentiment_analyzer(e.title[:128])[0]['label'].split()[0])
                        stars_sum += s
                        title = GoogleTranslator(source='en', target='ja').translate(e.title) if not is_j else e.title
                        news_list.append({"title": title, "score": s, "link": e.link})
                    avg_score = stars_sum / len(news_list)
                else: avg_score = 3.0
                
                sentiments_all.append(avg_score)
                adv, col = ("🌟【強気】", "#d4edda") if avg_score >= 3.5 and pred > curr else ("⚠️【警戒】", "#f8d7da") if avg_score <= 2.2 else ("😐【様子見】", "#e2e3e5")
                
                results_temp.append({
                    "銘柄": name, "将来": f_inv * (pred / curr), "adv": adv, "col": col, 
                    "news": news_list, "stars": avg_score, "gain": f_inv * (pred / curr) - f_inv, "pred_val": pred
                })
            except: continue

    st.session_state.results = results_temp
    st.session_state.plot_data = plot_data_temp
    
    # キャラ更新
    if sentiments_all:
        avg_v = sum(sentiments_all)/len(sentiments_all)
        if avg_v >= 3.7: st.session_state.char_msg = "AIの気分も最高だよ！攻めてみる？🚀"
        elif avg_v <= 2.3: st.session_state.char_msg = "うーん、少し怖いデータが出てるね…☔"
        else: st.session_state.char_msg = "分析完了！今はどっしり構えて良さそう☕"
    st.rerun()

# --- 7. 結果表示エリア ---
if st.session_state.results:
    st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
    
    # グラフ表示（予測ターゲットの星★を追加）
    fig, ax = plt.subplots(figsize=(10, 4))
    japanize_matplotlib.japanize()
    for res in st.session_state.results:
        name = res['銘柄']
        df = st.session_state.plot_data[name]
        base = df['Close'].iloc[0]
        # 成長率(%)でプロット
        line = ax.plot(df.index, df['Close']/base*100, label=name)
        # ★予測ターゲットプロット★
        ax.scatter(df.index[-1] + timedelta(days=5), (res['pred_val']/base)*100, 
                   marker='*', s=200, color=line[0].get_color(), edgecolors='black', label=f"{name} 予想地点", zorder=5)
    ax.set_ylabel("成長率 (%)")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)
    
    # 詳細カード
    for res in st.session_state.results:
        st.markdown(f"### 🎯 {res['銘柄']}")
        c_res1, c_res2 = st.columns([1, 2])
        c_res1.metric("予想額", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
        
        # 感情スコアと詳細（復活！）
        st.markdown(f"<div class='sentiment-badge'>AI感情分析値: {res['stars']:.1f} / 5.0 {'⭐' * int(res['stars'])}</div>", unsafe_allow_html=True)
        for n in res['news']:
            st.markdown(f"<div class='news-box'>{'★' * n['score']} <a href='{n['link']}' target='_blank'><b>{n['title']}</b></a></div>", unsafe_allow_html=True)

# 広告
st.markdown("""<div class="ad-container">
    <div class="ad-card">📊 証券口座なら<br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">DMM 株 口座開設 [PR]</a></div>
    <div class="ad-card">📱 投資アプリなら<br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">投資アプリ TOSSY [PR]</a></div>
</div>""", unsafe_allow_html=True)
