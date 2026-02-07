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
# 実際の運用時はここにご自身のデプロイしたURLを入力してください
APP_URL = "https://your-app-name.streamlit.app/" 
CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true"
INVESTMENT_QUOTES = [
    "「短期は感情、長期は理屈」だよ。今の相場はどう見える？",
    "「分散投資」は投資の世界で唯一のフリーランチ（タダ飯）なんだ。",
    "「木を見て森を見ず」にならないよう、広い視点で診断しよう！"
]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. セッション管理 ---
if "char_msg" not in st.session_state: st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
if "results" not in st.session_state: st.session_state.results = []
if "plot_data" not in st.session_state: st.session_state.plot_data = {}

# --- 3. CSS (ダークモード・レスポンシブ・各種装飾) ---
st.markdown(f"""
    <style>
    /* 全体フォント・サイズ調整 */
    html {{ font-size: 14px; }}
    @media (min-width: 768px) {{ html {{ font-size: 16px; }} }}

    /* 見出し装飾 */
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.1rem; border-left: 5px solid #3182ce; padding-left: 10px; margin: 20px 0 10px 0; }}
    
    /* 広告コンテナ (PCで横並び、スマホで縦並び) */
    .ad-row {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 30px 0; width: 100%; }}
    .ad-card {{ 
        flex: 1; min-width: 290px; padding: 20px; 
        border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 15px; 
        background: rgba(128, 128, 128, 0.05); text-align: center;
        display: flex; flex-direction: column; justify-content: space-between;
    }}
    .ad-card a {{ display: block; background: #3182ce; color: white !important; padding: 12px; border-radius: 8px; font-weight: bold; text-decoration: none; margin-top: 10px; }}

    /* X(Twitter)シェアボタン */
    .x-share-button {{
        display: inline-block; background-color: #000000; color: #ffffff !important; 
        padding: 12px 24px; border-radius: 30px; text-decoration: none; 
        font-weight: bold; font-size: 0.9rem; margin: 15px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2); transition: 0.3s;
    }}
    .x-share-button:hover {{ transform: scale(1.02); opacity: 0.9; }}

    /* 診断アドバイスボックス */
    .advice-box {{ padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; color: #1a202c; margin-bottom: 15px; border: 1px solid rgba(0,0,0,0.1); }}

    /* ニュースボックス */
    .news-box {{ padding: 10px; border-radius: 8px; border-left: 5px solid #3182ce; margin-bottom: 8px; background: rgba(128, 128, 128, 0.1); font-size: 0.9rem; }}

    /* 免責事項ボックス */
    .disclaimer-box {{ font-size: 0.8rem; padding: 20px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.2); margin-top: 50px; line-height: 1.6; background: rgba(128, 128, 128, 0.02); color: gray; }}

    /* 浮遊キャラクター */
    .floating-char-box {{ position: fixed; bottom: 20px; right: 20px; z-index: 99; pointer-events: none; }}
    .char-img {{ width: 100px; mix-blend-mode: multiply; filter: contrast(110%); animation: float 3s ease-in-out infinite; }}
    @media (min-width: 768px) {{ .char-img {{ width: 140px; }} }}
    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-10px); }} }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. 補助ロジック ---
def clean_stock_name(name):
    return re.sub(r'[^\w\s\.]', '', name).strip().split(' ')[0]

# --- 5. メイン画面表示 ---
st.title("🤖 AIマーケット総合診断 Pro")

# --- 💡 解説セクション ---
with st.expander("💡 感情指数と期間設定についての解説"):
    st.markdown("""
    ### 📊 感情指数（AIスコア）とは？
    AIが最新のニュースタイトルを読み取り、市場の「期待」や「不安」を1.0〜5.0で数値化したものです。
    * **⭐4.0以上**: ポジティブなニュースが多く、上昇の勢い（モメンタム）が強い状態。
    * **⭐2.0以下**: 警戒ニュースが多く、一時的な下落リスクがある状態。

    ### ⏳ 分析期間の選び方
    * **1週間・30日**: 直近の価格変動を重視します。短期トレードの参考に。
    * **1年・5年**: 企業の業績や安定性を重視します。積立・長期投資の参考に。
    * **全期間(Max)**: 上場来の全データを使い、その銘柄の「本質的な成長力」を測ります。
    """)

st.markdown("<div class='main-step'>STEP 1 & 2: 銘柄入力と条件設定</div>", unsafe_allow_html=True)

# --- 入力欄 (フリー入力復活) ---
c_in1, c_in2 = st.columns([2, 1])
input_symbols = c_in1.text_input("銘柄コードをカンマ区切りで入力 (例: NVDA, 7203.T, AAPL)", value="NVDA, 7203.T")
f_inv = c_in2.number_input("シミュレーション金額(円)", min_value=1000, value=100000, step=10000)

time_span = st.select_slider("分析期間（長期ほど成長力を重視します）", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="1年")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

# 実行
if st.button("🚀 AI診断スタート"):
    st.session_state.results = []
    symbol_list = [s.strip().upper() for s in input_symbols.split(",") if s.strip()]
    
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('市場データを多角的に解析中...'):
        for symbol in symbol_list:
            try:
                # 株価取得
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty:
                    st.warning(f"銘柄 '{symbol}' のデータが見つかりませんでした。")
                    continue
                
                # 予測（線形回帰）
                y = df['Close'].tail(20).values
                x = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                pred_val = float(model.predict([[len(y)+5]])[0])
                curr = float(df['Close'].iloc[-1])
                
                # 感情分析 (RSSニュース)
                news_list, stars_sum = [], 0
                news_url = f"https://news.google.com/rss/search?q={symbol}&hl=ja&gl=JP"
                feed = feedparser.parse(news_url)
                if feed.entries:
                    for e in feed.entries[:3]:
                        s = int(st.session_state.sentiment_analyzer(e.title[:128])[0]['label'].split()[0])
                        stars_sum += s
                        news_list.append({"title": e.title, "link": e.link, "score": s})
                    avg_score = stars_sum / len(news_list)
                else: avg_score = 3.0
                
                # 判定
                adv, col = ("🚀 強気", "#d4edda") if avg_score >= 3.2 and pred_val > curr else ("⚠️ 警戒", "#f8d7da") if avg_score <= 2.2 else ("☕ 様子見", "#e2e3e5")
                
                st.session_state.results.append({
                    "銘柄": symbol, "将来": f_inv * (pred_val / curr), "adv": adv, "col": col, 
                    "news": news_list, "stars": avg_score, "gain": f_inv * (pred_val / curr) - f_inv, 
                    "period": time_span, "invest": f_inv,
                    "pred_date": (df.index[-1] + timedelta(days=5)).strftime('%m/%d')
                })
                st.session_state.plot_data[symbol] = df
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
        s = res['銘柄']
        if s in st.session_state.plot_data:
            d = st.session_state.plot_data[s]
            ax.plot(d.index, d['Close'] / d['Close'].iloc[0] * 100, label=s)
    ax.set_ylabel("成長率 (%)")
    ax.legend()
    st.pyplot(fig)

    for res in st.session_state.results:
        st.markdown(f"### 🎯 {res['銘柄']} ({res['period']}分析)")
        c_res1, c_res2 = st.columns([1, 1])
        c_res1.metric(f"{res['pred_date']} 予想資産額", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']} (感情指数: ⭐{res['stars']:.1f})</div>", unsafe_allow_html=True)
        
        # Xシェアテキスト
        share_text = (
            f"📈 【AIマーケット診断】\n"
            f"━━━━━━━━━━━━━━\n"
            f"🎯 銘柄：{res['銘銘柄'] if '銘銘柄' in res else res['銘柄']}\n"
            f"🔍 期間：{res['period']}\n"
            f"💰 投資額：{res['invest']:,.0f}円\n"
            f"📢 判定：{res['adv']}\n"
            f"🚀 予想：{res['将来']:,.0f}円\n"
            f"━━━━━━━━━━━━━━\n"
            f"AIが市場のトレンドを解析！詳細はこちら 👇\n"
            f"{APP_URL}"
        )
        x_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"
        st.markdown(f'<a href="{x_url}" target="_blank" class="x-share-button">𝕏 この診断結果をポストする</a>', unsafe_allow_html=True)

        with st.expander("📰 根拠となった最新ニュース"):
            for n in res['news']:
                st.markdown(f"<div class='news-box'>⭐{n['score']} <a href='{n['link']}' target='_blank'>{n['title']}</a></div>", unsafe_allow_html=True)
        st.divider()

# --- 7. 広告・免責・キャラ ---
st.markdown(f"""
<div class="ad-row">
    <div class="ad-card">
        <div>
            <span style="background:#ff4b4b; color:white; padding:2px 8px; border-radius:5px; font-size:0.7rem; font-weight:bold;">PR</span>
            <p style="font-weight:bold; margin:10px 0;">DMM 株</p>
            <p style="font-size:0.85rem; opacity:0.8;">スマホで最短即日取引！1株から買える手軽さが人気です。</p>
        </div>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">無料で口座開設</a>
    </div>
    <div class="ad-card">
        <div>
            <span style="background:#ff4b4b; color:white; padding:2px 8px; border-radius:5px; font-size:0.7rem; font-weight:bold;">PR</span>
            <p style="font-weight:bold; margin:10px 0;">高機能チャート TOSSY</p>
            <p style="font-size:0.85rem; opacity:0.8;">AI予測と組み合わせて、より精度の高い投資判断をサポート。</p>
        </div>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">アプリを体験</a>
    </div>
</div>

<div class="disclaimer-box">
    <b>⚠️ 免責事項（重要）</b><br>
    本アプリで提供される株価予測、感情指数、および診断結果は、過去の市場データと公開されたニュースを独自のアルゴリズムおよびAIモデルで解析したものであり、<b>将来の運用成果を保証するものではありません。</b><br>
    ・株式投資には元本割れのリスクがあります。投資判断は、経済情勢や企業業績を考慮し、必ずご自身の責任で行ってください。<br>
    ・本アプリの利用により生じたいかなる損害についても、開発者は一切の責任を負いません。<br>
    ・提供されるニュース情報は遅延する場合や不正確な場合があります。最新情報は各金融機関等の公式サイトをご確認ください。<br>
    ※本サービスの一部にはアフィリエイト広告が含まれています。
</div>

<div class="floating-char-box">
    <div style="background:white; color:#1a202c; border:2px solid #3182ce; border-radius:12px; padding:8px; font-size:0.8rem; font-weight:bold; width:180px; text-align:center; margin-bottom:10px; pointer-events:auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        {st.session_state.char_msg}
    </div>
    <img src="{CHARACTER_URL}" class="char-img">
</div>
""", unsafe_allow_html=True)
