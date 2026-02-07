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

# 初心者向け投資の名言
INVESTMENT_QUOTES = [
    "「まずは生き残れ。儲けるのはそれからだ」",
    "「卵を一つのカゴに盛るな。分散が身を守るよ」",
    "「安く買って、高く売る。基本だけど難しいね」",
    "「投資は、自分自身の将来へのプレゼントだよ」"
]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. セッション管理 ---
if "char_msg" not in st.session_state:
    st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
if "results" not in st.session_state:
    st.session_state.results = []
if "plot_data" not in st.session_state:
    st.session_state.plot_data = {}

# --- 3. CSS：デザイン & ユーザビリティ ---
st.markdown(f"""
    <style>
    /* 全体デザイン */
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 15px; border-left: 5px solid #3182ce; padding-left: 10px; }}
    div[data-testid="stMetric"] {{ background-color: rgba(150, 150, 150, 0.05); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.2); }}
    
    /* 広告コンテナ（アフィリエイト用） */
    .ad-container {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 30px 0; }}
    .ad-card {{ 
        flex: 1; min-width: 280px; max-width: 450px; padding: 20px; 
        border: 2px dashed #cbd5e0; border-radius: 15px; text-align: center; 
        background-color: #f7fafc; transition: 0.3s;
    }}
    .ad-card:hover {{ border-color: #3182ce; background-color: #ebf8ff; }}
    .ad-card a {{ text-decoration: none; color: #3182ce; font-weight: bold; font-size: 1.1em; }}

    /* キャラクター固定配置 */
    .floating-char-box {{ position: fixed; bottom: 20px; right: 20px; z-index: 999; display: flex; flex-direction: column; align-items: center; pointer-events: none; }}
    .char-img {{ width: 140px; mix-blend-mode: multiply; filter: contrast(125%) brightness(108%); animation: float 3s ease-in-out infinite; }}
    .auto-quote-bubble {{
        background: white; border: 2px solid #3182ce; border-radius: 15px;
        padding: 10px 15px; margin-bottom: 10px; font-size: 0.85em; font-weight: bold; color: #1a202c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15); width: 220px; text-align: center; position: relative;
    }}
    .auto-quote-bubble::after {{ content: ""; position: absolute; bottom: -10px; right: 45%; border-width: 10px 10px 0; border-style: solid; border-color: #ffffff transparent; }}

    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-12px); }} }}
    
    /* 診断結果パーツ */
    .news-box {{ background: white; padding: 12px; border-radius: 8px; border-left: 5px solid #3182ce; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .advice-box {{ padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); }}
    .sentiment-badge {{ background: #3182ce; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; margin-bottom: 10px; }}
    .disclaimer-box {{ font-size: 0.8em; color: #718096; background: #f7fafc; padding: 20px; border-radius: 10px; margin-top: 50px; line-height: 1.6; border: 1px solid #e2e8f0; }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. 市場指標関数 ---
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

# --- 5. メイン表示 ---
st.title("🤖 AIマーケット総合診断 Pro")
st.markdown(f"""<div class="floating-char-box"><div class="auto-quote-bubble">{st.session_state.char_msg}</div><img src="{CHARACTER_URL}" class="char-img"></div>""", unsafe_allow_html=True)

# 指標表示
idx_data = get_market_indices()
m1, m2, m3 = st.columns(3)
if idx_data.get("ドル円"): m1.metric("💴 ドル/円", f"{idx_data['ドル円'][0]:,.2f}円", f"{idx_data['ドル円'][1]:+,.2f}")
if idx_data.get("日経平均"): m2.metric("🇯🇵 日経平均", f"{idx_data['日経平均'][0]:,.2f}円", f"{idx_data['日経平均'][1]:+,.2f}")
if idx_data.get("NYダウ"): m3.metric("🇺🇸 NYダウ", f"{idx_data['NYダウ'][0]:,.2f}ドル", f"{idx_data['NYダウ'][1]:+,.2f}")

st.markdown("---")

# 入力セクション
st.markdown("<div class='main-step'>STEP 1: 診断したい銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T"}
c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("よく選ばれる銘柄", list(stock_presets.keys()), default=["エヌビディア"])
free_input = c_in2.text_input("コードで直接入力 (例: MSFT, 9984.T)", "")
final_targets = {name: stock_presets[name] for name in selected_names}
if free_input: final_targets[free_input.upper()] = free_input.upper()

st.markdown("<div class='main-step'>STEP 2: 投資条件の確認</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
f_inv = c1.number_input("シミュレーションする金額(円)", min_value=1000, value=100000, step=10000)
time_span = c2.select_slider("過去の参照期間", options=["1週間", "30日", "1年", "5年"], value="30日")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y"}

if st.button("🚀 AI診断を開始する"):
    results_temp, plot_data_temp = [], {}
    sentiments_all = []
    
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('最新ニュースと株価データを照らし合わせています...'):
        for name, symbol in final_targets.items():
            try:
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                
                # AI予測（線形回帰）
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model = LinearRegression().fit(X_reg, y_reg)
                pred_val = float(model.predict([[len(y_reg)+5]])[0][0])
                
                # ニュース感情分析
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
                adv, col = ("🚀 強気判定", "#d4edda") if avg_score >= 3.5 and pred_val > curr else ("⚠️ 警戒判定", "#f8d7da") if avg_score <= 2.2 else ("☕ 様子見", "#e2e3e5")
                
                plot_data_temp[name] = df
                results_temp.append({
                    "銘柄": name, "将来": f_inv * (pred_val / curr), "adv": adv, "col": col, 
                    "news": news_list, "stars": avg_score, "gain": f_inv * (pred_val / curr) - f_inv, 
                    "pred_val": pred_val, "curr_val": curr
                })
            except: continue

    st.session_state.results = results_temp
    st.session_state.plot_data = plot_data_temp
    
    # AIキャラの反応
    if sentiments_all:
        avg_v = sum(sentiments_all)/len(sentiments_all)
        if avg_v >= 3.7: st.session_state.char_msg = "分析完了！ポジティブなニュースが多いね。ワクワクするよ！🚀"
        elif avg_v <= 2.3: st.session_state.char_msg = "少し厳しいニュースがあるみたい…今は慎重にいこうね☔"
        else: st.session_state.char_msg = "結果が出たよ。落ち着いた市場環境みたいだね。じっくり見守ろう☕"
    st.rerun()

# --- 7. 診断結果の表示エリア ---
if st.session_state.results:
    st.markdown("<div class='main-step'>STEP 3: AIの診断結果</div>", unsafe_allow_html=True)
    
    # グラフ：将来の予想地点に★マーク
    fig, ax = plt.subplots(figsize=(10, 4))
    japanize_matplotlib.japanize()
    for res in st.session_state.results:
        name = res['銘柄']
        if name in st.session_state.plot_data:
            df = st.session_state.plot_data[name]
            base = df['Close'].iloc[0]
            line = ax.plot(df.index, df['Close']/base*100, label=f"{name} (実績)", linewidth=2)
            # 星マーク予測
            p_val = res.get('pred_val')
            if p_val:
                ax.scatter(df.index[-1] + timedelta(days=5), (p_val/base)*100, 
                           marker='*', s=250, color=line[0].get_color(), edgecolors='black', label=f"{name} 5日後予想", zorder=5)
    ax.set_title("銘柄ごとの成長予測（開始時を100とした場合）", fontsize=12)
    ax.set_ylabel("成長率 (%)")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    # 感情指標の解説
    with st.expander("💡 「AI感情分析値（星の数）」の見かたについて"):
        st.write("""
        AIが最新のニュース記事を読み、その内容が「ポジティブ」か「ネガティブ」かを判定しています。
        - ⭐⭐⭐⭐⭐ (5.0): 非常に良いニュースが多い状態。期待が高まっています。
        - ⭐⭐⭐ (3.0): ニュースが少ない、または良い悪いが混ざっている中立な状態。
        - ⭐ (1.0): 厳しい決算や社会的な懸念など、ネガティブな材料が多い状態。
        """)

    # 銘柄別の詳細カード
    for res in st.session_state.results:
        st.markdown(f"### 🎯 {res['銘柄']} の分析結果")
        c_res1, c_res2 = st.columns([1, 2])
        c_res1.metric("5日後の予想資産額", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='sentiment-badge'>AI感情分析値: {res['stars']:.1f} / 5.0 {'⭐' * int(res['stars'])}</div>", unsafe_allow_html=True)
        for n in res['news']:
            st.markdown(f"<div class='news-box'>{'★' * n['score']} <a href='{n['link']}' target='_blank'><b>{n['title']}</b></a></div>", unsafe_allow_html=True)

# --- 8. 広告 & 免責事項 ---
st.markdown("""<div class="ad-container">
    <div class="ad-card">
        <p>📊 初心者に人気の証券口座</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">DMM 株 口座開設はこちら [PR]</a>
    </div>
    <div class="ad-card">
        <p>📱 投資をスマホでもっと手軽に</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">投資アプリ TOSSY [PR]</a>
    </div>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="disclaimer-box">
        <strong>【免責事項・必ずお読みください】</strong><br>
        ● 本アプリはAI技術を用いた情報の提供を目的としており、特定の銘柄への投資を勧誘するものではありません。診断結果は将来の成果を保証するものではなく、実際の市場では予期せぬ変動が起こる可能性があります。<br>
        ● 投資の最終決定は、必ずご自身の判断と責任において行ってください。本アプリの利用によって生じた損失や損害について、提供者は一切の責任を負いかねます。<br>
        ● 本ページにはアフィリエイト広告が含まれており、紹介しているサービスへのリンクを通じて報酬を得る場合があります。診断結果の透明性には配慮しておりますが、広告主の影響を完全に排除したものではありません。<br>
        ● 計画的な投資と、生活に支障のない範囲での余剰資金による運用を強くおすすめします。
    </div>
    """, unsafe_allow_html=True)
