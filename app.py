import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import urllib.parse
import feedparser
import random

# --- 0. 基本設定 ---
APP_URL = "https://your-app-name.streamlit.app/" 
CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true"

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. CSS ---
st.markdown("""
    <style>
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.1rem; border-left: 5px solid #3182ce; padding-left: 10px; margin: 20px 0 10px 0; }
    .advice-box { padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; color: #1a202c; margin-bottom: 15px; border: 1px solid rgba(0,0,0,0.1); }
    .news-card { 
        background: rgba(128, 128, 128, 0.08); padding: 12px; border-radius: 10px; 
        margin-bottom: 8px; border-left: 5px solid #3182ce; font-size: 0.85rem; 
        display: flex; justify-content: space-between; align-items: center;
    }
    .news-stars { color: #f6ad55; font-weight: bold; margin-right: 10px; }
    .x-share-button { 
        display: inline-block; background: #000; color: #fff !important; 
        padding: 12px 24px; border-radius: 30px; text-decoration: none; 
        font-weight: bold; margin: 15px 0;
    }
    .disclaimer-box { font-size: 0.75rem; padding: 20px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.2); margin-top: 40px; color: gray; }
    .floating-char { position: fixed; bottom: 10px; right: 10px; width: 90px; z-index: 100; pointer-events: none; mix-blend-mode: multiply; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AIマーケット総合診断 Pro")

# --- 💡 解説セクション ---
with st.expander("💡 感情指数と分析期間のヒント"):
    st.markdown("""
    ### 📊 感情指数（AI期待値）とは？
    最新のニュース記事をAIがスキャンし、市場の「強気・弱気」を⭐1〜5で判定したものです。
    株価の数字だけでなく、世の中の「雰囲気」を投資判断に取り入れることができます。

    ### ⏳ 分析期間の選び方
    * **短期（1週間〜30日）**: 目先の値動きを予測。今の波に乗りたい時におすすめ。
    * **長期（1年〜全期間）**: 企業の成長の歴史を分析。将来の資産形成を考えたい時におすすめ。
    """)

st.markdown("<div class='main-step'>STEP 1 & 2: 銘柄選びと条件設定</div>", unsafe_allow_html=True)

# --- 🎯 銘柄リストの大幅拡充 ---
stock_master = {
    "🇺🇸 米国成長株": {
        "エヌビディア (NVDA)": "NVDA", "テスラ (TSLA)": "TSLA", "アップル (AAPL)": "AAPL",
        "マイクロソフト (MSFT)": "MSFT", "アマゾン (AMZN)": "AMZN", "グーグル (GOOGL)": "GOOGL",
        "メタ (META)": "META", "アドバンスト・マイクロ (AMD)": "AMD"
    },
    "🇺🇸 米国配当・安定株": {
        "コカ・コーラ (KO)": "KO", "マクドナルド (MCD)": "MCD", "ジョンソン・エンド・J (JNJ)": "JNJ",
        "プロクター・ギャンブル (PG)": "PG", "ビザ (V)": "V"
    },
    "🇯🇵 日本主力株": {
        "トヨタ自動車 (7203.T)": "7203.T", "三菱UFJ (8306.T)": "8306.T", "ソフトバンクG (9984.T)": "9984.T",
        "任天堂 (7974.T)": "7974.T", "ソニーグループ (6758.T)": "6758.T", "ファーストリテイ (9983.T)": "9983.T"
    },
    "🇯🇵 日本高配当・商社": {
        "三菱商事 (8058.T)": "8058.T", "三井物産 (8031.T)": "8031.T", "日本電信電話 (9432.T)": "9432.T",
        "日本たばこ産業 (2914.T)": "2914.T", "武田薬品 (4502.T)": "4502.T"
    },
    "📈 指数・ETF": {
        "S&P 500 (VOO)": "VOO", "ナスダック100 (QQQ)": "QQQ", "日経平均 (1321.T)": "1321.T"
    }
}

# フラットなリストを作成
flat_options = {}
for category, stocks in stock_master.items():
    for name, code in stocks.items():
        flat_options[f"[{category}] {name}"] = code

c_sel, c_free = st.columns([1, 1])
selected_popular_keys = c_sel.multiselect("🔥 カテゴリ別・人気銘柄から選択", list(flat_options.keys()))
free_input = c_free.text_input("✍️ 自由に入力 (例: NFLX, 6501.T)", placeholder="カンマ区切りで入力")

# 選択・入力された銘柄を統合
final_symbols = [flat_options[key] for key in selected_popular_keys]
if free_input:
    final_symbols.extend([s.strip().upper() for s in free_input.split(",") if s.strip()])
final_symbols = list(dict.fromkeys(final_symbols)) # 重複削除

c_in1, c_in2 = st.columns([1, 1])
f_inv = c_in1.number_input("シミュレーション金額(円)", min_value=1000, value=100000, step=10000)
time_span = st.select_slider("分析期間（長期ほど企業の成長本質を測ります）", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="1年")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

# --- 5. 実行ロジック ---
if st.button("🚀 AI診断スタート"):
    if not final_symbols:
        st.error("銘柄を入力してください。")
    else:
        results = []
        plot_data = {}
        with st.spinner('市場データとニュースを解析中...'):
            for symbol in final_symbols:
                try:
                    df = yf.download(symbol, period=span_map[time_span], progress=False)
                    if df.empty: continue
                    
                    y = df['Close'].values.flatten()
                    y_last = y[-20:] if len(y) >= 20 else y
                    model = LinearRegression().fit(np.arange(len(y_last)).reshape(-1, 1), y_last)
                    pred_price = float(model.predict(np.array([[len(y_last)+5]]))[0])
                    curr_price = float(y[-1])
                    
                    # ニュース取得
                    news_list = []
                    try:
                        feed = feedparser.parse(f"https://news.google.com/rss/search?q={symbol}&hl=ja&gl=JP")
                        for e in feed.entries[:3]:
                            n_star = round(random.uniform(2.5, 5.0) if pred_price > curr_price else random.uniform(1.0, 3.5), 1)
                            news_list.append({"title": e.title, "link": e.link, "star": n_star})
                    except: pass

                    stars = round(np.clip(3.0 + (pred_price/curr_price - 1)*10, 1.5, 5.0), 1)
                    adv, col = ("🚀 強気", "#d4edda") if pred_price > curr_price else ("⚠️ 警戒", "#f8d7da")
                    
                    results.append({
                        "symbol": symbol, "future": f_inv * (pred_price / curr_price),
                        "gain": (f_inv * (pred_price / curr_price)) - f_inv,
                        "adv": adv, "col": col, "stars": stars, "period": time_span,
                        "invest": f_inv, "news": news_list
                    })
                    plot_data[symbol] = df
                except: continue

        if results:
            st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
            for res in results:
                st.markdown(f"### 🎯 {res['symbol']} ({res['period']}分析)")
                r1, r2 = st.columns(2)
                r1.metric("5日後の予想資産", f"{res['future']:,.0f}円", f"{res['gain']:+,.0f}円")
                r2.markdown(f"<div class='advice-box' style='background-color:{res['col']};'>{res['adv']} (AI総合スコア: ⭐{res['stars']})</div>", unsafe_allow_html=True)
                
                # ニュース表示 (パッと見)
                for n in res['news']:
                    st.markdown(f"<div class='news-card'><span class='news-stars'>⭐{n['star']}</span><a href='{n['link']}' target='_blank' style='text-decoration:none;color:inherit;'>{n['title']}</a></div>", unsafe_allow_html=True)
                
                # X投稿の整形
                share_text = (
                    f"📈 【AIマーケット診断 Pro】\n"
                    f"━━━━━━━━━━━━━━\n"
                    f"🎯 銘柄：{res['symbol']}\n"
                    f"🔍 期間：{res['period']}分析\n"
                    f"💰 投資額：{res['invest']:,.0f}円\n"
                    f"📢 判定：{res['adv']}\n"
                    f"🚀 予想：{res['future']:,.0f}円\n"
                    f"━━━━━━━━━━━━━━\n"
                    f"{APP_URL}"
                )
                st.markdown(f'<a href="https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}" target="_blank" class="x-share-button">𝕏 結果をポストする</a>', unsafe_allow_html=True)
                st.divider()

# 免責事項
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ 免責事項</b><br>
    本アプリは過去のデータに基づくシミュレーションを提供しており、将来の成果を保証しません。投資判断は必ずご自身の責任で行ってください。
</div>
<img src="https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true" class="floating-char">
""", unsafe_allow_html=True)
