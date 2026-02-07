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
    "「木を見て森を見ず」にならないように、期間を変えてチェックしよう！",
    "「短期は感情、長期は理屈」で動くのが相場の常だよ。",
    "「どの期間で戦うか」を決めることが、投資の第一歩だね。"
]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro (Max版)", layout="wide", page_icon="📈")

# --- 2. セッション管理 ---
if "char_msg" not in st.session_state:
    st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
if "results" not in st.session_state:
    st.session_state.results = []
if "plot_data" not in st.session_state:
    st.session_state.plot_data = {}

# --- 3. CSS（デザイン調整） ---
st.markdown(f"""
    <style>
    .welcome-box {{
        background-color: #f0f7ff; padding: 20px; border-radius: 15px; 
        border: 1px solid #3182ce; margin-bottom: 25px;
    }}
    .feature-tag {{
        background: #3182ce; color: white; padding: 2px 10px; 
        border-radius: 5px; font-size: 0.8em; margin-right: 5px;
    }}
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 15px; border-left: 5px solid #3182ce; padding-left: 10px; }}
    .ad-container {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 30px 0; }}
    .ad-card {{ flex: 1; min-width: 280px; max-width: 450px; padding: 20px; border: 2px dashed #cbd5e0; border-radius: 15px; text-align: center; background-color: #f7fafc; }}
    .ad-card a {{ text-decoration: none; color: #3182ce; font-weight: bold; }}
    .floating-char-box {{ position: fixed; bottom: 20px; right: 20px; z-index: 999; display: flex; flex-direction: column; align-items: center; pointer-events: none; }}
    .char-img {{ width: 140px; mix-blend-mode: multiply; filter: contrast(125%) brightness(108%); animation: float 3s ease-in-out infinite; }}
    .auto-quote-bubble {{ background: white; border: 2px solid #3182ce; border-radius: 15px; padding: 10px 15px; margin-bottom: 10px; font-size: 0.85em; font-weight: bold; width: 220px; text-align: center; position: relative; }}
    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-12px); }} }}
    .advice-box {{ padding: 20px; border-radius: 15px; text-align: center; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); }}
    .sentiment-badge {{ background: #3182ce; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; margin-bottom: 10px; }}
    .disclaimer-box {{ font-size: 0.8em; color: #718096; background: #f7fafc; padding: 20px; border-radius: 10px; margin-top: 50px; border: 1px solid #e2e8f0; }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. 市場指標取得 ---
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

# --- 5. メイン画面 ---
st.title("🤖 AIマーケット総合診断 Pro (Max)")

# 【追加】初めての人向けガイド
st.markdown("""
<div class="welcome-box">
    <h4 style="margin-top:0;">🌟 はじめての方へ：このアプリでできること</h4>
    <p style="font-size:0.95em;">
        「今の株価」だけでなく、<b>AIがあなたの投資を多角的にサポート</b>する次世代診断ツールです。
    </p>
    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        <div><span class="feature-tag">予測</span> <b>1. 未来を予測</b>：過去のトレンドから5日後の株価をAIが算出します。</div>
        <div><span class="feature-tag">分析</span> <b>2. ニュースを星で評価</b>：最新記事をAIが読み取り、世の中の期待度を星5段階で表示。</div>
        <div><span class="feature-tag">視点</span> <b>3. 歴史を振り返る</b>：全期間のチャートから、その銘柄の真の実力を確認できます。</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""<div class="floating-char-box"><div class="auto-quote-bubble">{st.session_state.char_msg}</div><img src="{CHARACTER_URL}" class="char-img"></div>""", unsafe_allow_html=True)

# 指標表示
idx_data = get_market_indices()
cols = st.columns(3)
for i, (k, v) in enumerate(idx_data.items()):
    if v and v[0]: cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:+,.2f}")

st.markdown("---")

# 星の指標解説
with st.expander("⭐ 「星の指標（AI感情分析）」の詳細な見かた"):
    st.write("""
    最新ニュースから「市場の熱気」をAIが数値化したものです。
    - **5.0 (⭐⭐⭐⭐⭐)**: 絶好調。ポジティブなニュースで溢れています。
    - **3.0 (⭐⭐⭐)**: 普通。良い材料も悪い材料も混在しています。
    - **1.0 (⭐)**: 警戒。厳しいニュースや懸念材料が目立ちます。
    """)

# STEP 1: 銘柄選択
st.markdown("<div class='main-step'>STEP 1: 診断したい銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "トヨタ": "7203.T", "ソニー": "6758.T"}
c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("よく選ばれる銘柄", list(stock_presets.keys()), default=["エヌビディア"])
free_input = c_in2.text_input("直接入力 (例: MSFT, 9984.T)", "")
final_targets = {name: stock_presets[name] for name in selected_names}
if free_input: final_targets[free_input.upper()] = free_input.upper()

# STEP 2: 設定
st.markdown("<div class='main-step'>STEP 2: 分析条件を設定しよう</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
f_inv = c1.number_input("投資シミュレーション金額(円)", min_value=1000, value=100000)
time_span = c2.select_slider("参照期間（期間によって結果が変わります）", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="全期間(Max)")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

# --- 6. 診断実行 ---
if st.button("🚀 AI診断スタート"):
    results_temp, plot_data_temp = [], {}
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('AIが歴史と感情を分析中...'):
        for name, symbol in final_targets.items():
            try:
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                
                # 予測計算
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model = LinearRegression().fit(X_reg, y_reg)
                pred_val = float(model.predict([[len(y_reg)+5]])[0][0])
                pred_date = (df.index[-1] + timedelta(days=5)).strftime('%m/%d')
                
                # ニュース取得
                q = name if ".T" in symbol else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ja&gl=JP"
                feed = feedparser.parse(url)
                news_list, stars_sum = [], 0
                if feed.entries:
                    for e in feed.entries[:3]:
                        s = int(st.session_state.sentiment_analyzer(e.title[:128])[0]['label'].split()[0])
                        stars_sum += s
                        title = GoogleTranslator(source='en', target='ja').translate(e.title) if ".T" not in symbol else e.title
                        news_list.append({"title": title, "score": s, "link": e.link})
                    avg_score = stars_sum / len(news_list)
                else: avg_score = 3.0
                
                adv, col = ("🚀 強気判定", "#d4edda") if avg_score >= 3.5 and pred_val > curr else ("⚠️ 警戒判定", "#f8d7da") if avg_score <= 2.2 else ("☕ 様子見", "#e2e3e5")
                
                plot_data_temp[name] = df
                results_temp.append({
                    "銘柄": name, "将来": f_inv * (pred_val / curr), "adv": adv, "col": col, 
                    "news": news_list, "stars": avg_score, "gain": f_inv * (pred_val / curr) - f_inv, 
                    "pred_val": pred_val, "pred_date": pred_date, "period_label": time_span
                })
            except: continue

    st.session_state.results = results_temp
    st.session_state.plot_data = plot_data_temp
    st.rerun()

# --- 7. 結果表示 ---
if st.session_state.results:
    first_res = st.session_state.results[0]
    display_label = first_res.get('period_label', '選択期間')
    st.markdown(f"<div class='main-step'>STEP 3: {display_label}の診断結果</div>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    japanize_matplotlib.japanize()
    for res in st.session_state.results:
        name = res['銘柄']
        if name in st.session_state.plot_data:
            df = st.session_state.plot_data[name]
            base = df['Close'].iloc[0]
            line = ax.plot(df.index, df['Close']/base*100, label=f"{name}")
            p_val = res.get('pred_val')
            p_date = res.get('pred_date', '将来')
            if p_val:
                ax.scatter(df.index[-1] + timedelta(days=5), (p_val/base)*100, 
                           marker='*', s=250, color=line[0].get_color(), edgecolors='black', label=f"{name} {p_date}予想", zorder=5)
    ax.set_ylabel("成長率 (%)")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    for res in st.session_state.results:
        st.markdown(f"### 🎯 {res['銘柄']}")
        c_res1, c_res2 = st.columns([1, 2])
        p_date_str = res.get('pred_date', '5日後')
        c_res1.metric(f"{p_date_str} の予想資産額", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sentiment-badge'>AI感情分析: {res.get('stars', 3.0):.1f} / 5.0 {'⭐' * int(res.get('stars', 3))}</div>", unsafe_allow_html=True)
        for n in res.get('news', []):
            st.markdown(f"<div class='news-box'>{'★' * n['score']} <a href='{n['link']}' target='_blank'><b>{n['title']}</b></a></div>", unsafe_allow_html=True)

# 広告 & 免責
st.markdown("""<div class="ad-container">
    <div class="ad-card"><p>📊 証券口座なら</p><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">DMM 株 口座開設はこちら [PR]</a></div>
    <div class="ad-card"><p>📱 投資アプリなら</p><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">投資アプリ TOSSY [PR]</a></div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="disclaimer-box"><strong>【免責事項】</strong><br>● 予想額は過去のトレンドに基づくシミュレーションであり、将来の成果を保証するものではありません。● 期間設定により診断結果は大きく変わる場合があります。● 「星の指標」はAIによる感情分析結果であり、投資勧誘ではありません。● 最終的な判断は必ずご自身で行ってください。[PR]報酬を得る場合があります。</div>""", unsafe_allow_html=True)
