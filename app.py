import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd  # ← ここを修正しました（pdではなくpandasをimport）
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
CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true"
INVESTMENT_QUOTES = [
    "「木を見て森を見ず」にならないように、期間を変えてチェックしよう！",
    "「短期は感情、長期は理屈」で動くのが相場の常だよ。",
    "「分散投資」は、投資の世界で唯一のフリーランチ（タダ飯）だよ。"
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

# --- 3. CSS (Xシェアボタンとデザイン) ---
st.markdown(f"""
    <style>
    .welcome-box {{ background-color: #f0f7ff; padding: 20px; border-radius: 15px; border: 1px solid #3182ce; margin-bottom: 25px; }}
    .feature-tag {{ background: #3182ce; color: white; padding: 2px 10px; border-radius: 5px; font-size: 0.8em; margin-right: 5px; }}
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 15px; border-left: 5px solid #3182ce; padding-left: 10px; }}
    .x-share-button {{
        display: inline-block; background-color: #000000; color: white !important; 
        padding: 8px 18px; border-radius: 20px; text-decoration: none; 
        font-weight: bold; font-size: 0.85em; margin-top: 10px; border: none;
        transition: 0.3s;
    }}
    .x-share-button:hover {{ background-color: #333333; opacity: 0.9; }}
    .floating-char-box {{ position: fixed; bottom: 20px; right: 20px; z-index: 999; display: flex; flex-direction: column; align-items: center; pointer-events: none; }}
    .char-img {{ width: 140px; mix-blend-mode: multiply; filter: contrast(125%) brightness(108%); animation: float 3s ease-in-out infinite; }}
    .auto-quote-bubble {{ background: white; border: 2px solid #3182ce; border-radius: 15px; padding: 10px 15px; margin-bottom: 10px; font-size: 0.85em; font-weight: bold; width: 220px; text-align: center; position: relative; }}
    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-12px); }} }}
    .advice-box {{ padding: 20px; border-radius: 15px; text-align: center; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); }}
    .sentiment-badge {{ background: #3182ce; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; margin-bottom: 10px; }}
    .news-box {{ background: white; padding: 10px; border-radius: 8px; border-left: 5px solid #3182ce; margin-bottom: 8px; font-size: 0.9em; }}
    .disclaimer-box {{ font-size: 0.8em; color: #718096; background: #f7fafc; padding: 20px; border-radius: 10px; margin-top: 50px; border: 1px solid #e2e8f0; }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. 銘柄リスト & 名前クリーンアップ ---
STOCK_PRESETS = {
    "🇺🇸 エヌビディア (AI半導体)": "NVDA", "🇺🇸 テスラ (電気自動車)": "TSLA", "🇺🇸 アップル (iPhone)": "AAPL",
    "🇺🇸 マイクロソフト (AI/OS)": "MSFT", "🇺🇸 アマゾン (EC)": "AMZN", "🇺🇸 アルファベット (Google)": "GOOGL",
    "🇯🇵 トヨタ自動車 (世界一)": "7203.T", "🇯🇵 ソニーG (エンタメ)": "6758.T", "🇯🇵 ソフトバンクG (投資)": "9984.T",
    "🇯🇵 任天堂 (ゲーム)": "7974.T", "🇯🇵 三菱UFJ銀 (金融)": "8306.T", "🇯🇵 キーエンス (高収益)": "6861.T"
}

def clean_stock_name(name):
    name = re.sub(r'[^\w\s\.]', '', name)
    return name.strip().split(' ')[0]

# --- 5. メイン表示 ---
st.title("🤖 AIマーケット総合診断 Pro (Max)")

st.markdown("""
<div class="welcome-box">
    <h4 style="margin-top:0;">🌟 はじめての方へ：このアプリでできること</h4>
    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        <div><span class="feature-tag">予測</span> <b>1. 未来予測</b>：5日後の株価をAI算出。</div>
        <div><span class="feature-tag">分析</span> <b>2. 星判定</b>：最新ニュースを星5段階で判定。</div>
        <div><span class="feature-tag">共有</span> <b>3. Xでポスト</b>：診断結果をX(Twitter)に投稿可能！</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("💡 分析期間を変えると結果が変わるのはなぜ？"):
    st.write("""
    投資の目的（ゴール）によって、AIが見るべきデータが異なるからです。
    - **「1週間/30日」を選んだ場合**: AIは「今の勢い（トレンド）」を重視します。短期的な投資の参考になります。
    - **「5年/全期間」を選んだ場合**: AIは「その銘柄が本来持っている成長力」を重視します。長期的な資産形成の参考になります。
    """)

with st.expander("⭐ 「星の指標（AI感情分析）」とは？"):
    st.write("最新ニュースをAIが読み取り、1.0〜5.0で数値化したものです。5に近いほど期待が高まっています。")

st.markdown(f"""<div class="floating-char-box"><div class="auto-quote-bubble">{st.session_state.char_msg}</div><img src="{CHARACTER_URL}" class="char-img"></div>""", unsafe_allow_html=True)

# STEP 1 & 2
st.markdown("<div class='main-step'>STEP 1 & 2: 診断したい銘柄と条件を選ぼう</div>", unsafe_allow_html=True)
c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("リストから選ぶ", list(STOCK_PRESETS.keys()), default=["🇺🇸 エヌビディア (AI半導体)"])
f_inv = c_in2.number_input("シミュレーション金額(円)", min_value=1000, value=100000, step=10000)

time_span = st.select_slider("分析する期間を選択", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="全期間(Max)")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

# 実行
if st.button("🚀 AI診断スタート"):
    results_temp, plot_data_temp = [], {}
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('データを解析中...'):
        for full_name in selected_names:
            try:
                symbol = STOCK_PRESETS[full_name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model = LinearRegression().fit(X_reg, y_reg)
                pred_val = float(model.predict([[len(y_reg)+5]])[0][0])
                pred_date = (df.index[-1] + timedelta(days=5)).strftime('%m/%d')
                
                display_name = clean_stock_name(full_name)
                
                q = display_name if ".T" in symbol else symbol
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
                
                adv, col = ("🚀 強気", "#d4edda") if avg_score >= 3.5 and pred_val > curr else ("⚠️ 警戒", "#f8d7da") if avg_score <= 2.2 else ("☕ 様子見", "#e2e3e5")
                
                plot_data_temp[display_name] = df
                results_temp.append({
                    "銘柄": display_name, "将来": f_inv * (pred_val / curr), "adv": adv, "col": col, 
                    "news": news_list, "stars": avg_score, "gain": f_inv * (pred_val / curr) - f_inv, 
                    "pred_val": pred_val, "pred_date": pred_date, "period_label": time_span
                })
            except: continue

    st.session_state.results = results_temp
    st.session_state.plot_data = plot_data_temp
    st.rerun()

# --- 7. 結果表示 ---
if st.session_state.results:
    st.markdown(f"<div class='main-step'>STEP 3: {st.session_state.results[0].get('period_label')}の診断結果</div>", unsafe_allow_html=True)
    
    # グラフ
    fig, ax = plt.subplots(figsize=(10, 4))
    japanize_matplotlib.japanize()
    for res in st.session_state.results:
        name = res['銘柄']
        if name in st.session_state.plot_data:
            df = st.session_state.plot_data[name]
            base = df['Close'].iloc[0]
            line = ax.plot(df.index, df['Close']/base*100, label=f"{name}")
            ax.scatter(df.index[-1] + timedelta(days=5), (res['pred_val']/base)*100, marker='*', s=200, color=line[0].get_color(), edgecolors='black', zorder=5)
    ax.set_ylabel("成長率 (%)")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    # 銘柄別詳細
    for res in st.session_state.results:
        st.markdown(f"### 🎯 {res['銘柄']}")
        c_res1, c_res2 = st.columns([1, 2])
        c_res1.metric(f"{res['pred_date']} 予想額", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
        
        # Xシェアボタン
        share_text = f"【AI株診断】\n銘柄: {res['銘柄']}\n判定: {res['adv']}\n5日後の予想: {res['将来']:,.0f}円！\n#AIマーケット診断"
        x_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"
        st.markdown(f'<a href="{x_url}" target="_blank" class="x-share-button">𝕏 この結果をポストする</a>', unsafe_allow_html=True)

        st.markdown(f"<div class='sentiment-badge'>AI感情分析: {res['stars']:.1f} / 5.0 {'⭐' * int(res['stars'])}</div>", unsafe_allow_html=True)
        for n in res['news']:
            st.markdown(f"<div class='news-box'>{'★' * n['score']} <a href='{n['link']}' target='_blank'><b>{n['title']}</b></a></div>", unsafe_allow_html=True)

# 広告・免責
st.markdown("""<div class="ad-container"><div class="ad-card"><p>📊 証券口座なら</p><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">DMM 株 [PR]</a></div><div class="ad-card"><p>📱 投資アプリなら</p><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">TOSSY [PR]</a></div></div>""", unsafe_allow_html=True)
st.markdown("<div class='disclaimer-box'>【免責】予想額は過去のトレンドに基づくシミュレーションであり、将来を保証しません。最終的な投資判断は必ずご自身で行ってください。[PR]</div>", unsafe_allow_html=True)
