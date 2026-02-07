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

# --- 0. グラフ表示の安定化設定 (これを追加) ---
import matplotlib
matplotlib.use('Agg')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AI投資診断アプリ", layout="wide")
st.title("🚀 AI銘柄診断ランキング")

# --- 2. AIモデルの読み込み ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

analyzer = load_ai()

# --- 3. 銘柄設定 ---
stocks = {
    "テスラ": "TSLA",
    "パランティア": "PLTR",
    "トヨタ": "7203.T",
    "任天堂": "7974.T",
    "エヌビディア": "NVDA",
    "Apple": "AAPL"
}
selected_names = st.sidebar.multiselect("分析する銘柄を選択", list(stocks.keys()), default=["テスラ", "パランティア", "トヨタ"])

# --- 4. 実行ボタン ---
if st.sidebar.button("分析を実行"):
    results = []
    plot_data = {} # グラフ用データを保存する場所
    
    with st.spinner('AIが分析中...'):
        for name in selected_names:
            try:
                symbol = stocks[name]
                df = yf.download(symbol, period="3mo", progress=False)
                if len(df) < 10: continue
                
                # グラフ用データの保存
                plot_data[name] = df.tail(30)

                # 予測計算
                y = df.tail(30)['Close'].values.reshape(-1, 1)
                X = np.arange(len(y)).reshape(-1, 1)
                
                model = LinearRegression()
                model.fit(X, y)
                pred_price = model.predict([[len(y)]])[0][0]
                
                last_price = float(y[-1][0])
                diff_pct = ((pred_price - last_price) / last_price) * 100
                
                # ニュース分析
                query = urllib.parse.quote(name)
                url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
                feed = feedparser.parse(url)
                
                stars, count = 0, 0
                for entry in feed.entries[:2]:
                    res = analyzer(entry.title)[0]
                    stars += int(res['label'].split()[0])
                    count += 1
                avg_stars = stars / count if count > 0 else 3
                
                results.append({
                    "銘柄": name,
                    "現在価格": round(last_price, 2),
                    "AI予測(明日)": round(float(pred_price), 2),
                    "期待値(%)": round(float(diff_pct), 2),
                    "ニュース評価": f"{avg_stars:.1f} ★",
                    "score": float(diff_pct) + (avg_stars - 3)
                })
            except Exception as e:
                continue

    if results:
        # ランキング表示
        res_df = pd.DataFrame(results).sort_values(by="score", ascending=False)
        res_df.insert(0, "順位", range(1, len(res_df) + 1))
        st.subheader("🏆 注目銘柄ランキング")
        st.table(res_df.drop(columns="score"))

        # グラフ表示
        st.subheader("📈 トレンド比較 (直近30日)")
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, data in plot_data.items():
            norm_price = data['Close'] / data['Close'].iloc[0] * 100
            ax.plot(data.index, norm_price, label=name)
        plt.axhline(100, color='black', linestyle='--', alpha=0.3)
        plt.legend()
        st.pyplot(fig)
    else:
        st.error("データを取得できませんでした。サイドバーで銘柄を選び直してください。")

st.info("💡 AI予測は統計的なトレンドに基づいています。投資判断は自己責任でお願いします。")
