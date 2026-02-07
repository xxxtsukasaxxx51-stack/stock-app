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

# --- 1. ページ設定 (アプリの見た目) ---
st.set_page_config(page_title="AI投資診断アプリ", layout="wide")

st.title("🚀 AI銘柄診断ランキング")
st.markdown("最新ニュースの感情分析と統計モデルによる、明日への投資ガイド。")

# --- 2. AIモデルの読み込み (キャッシュ機能で高速化) ---
@st.cache_resource
def load_ai():
    # 多言語対応の感情分析モデル
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

analyzer = load_ai()

# --- 3. サイドバー設定 ---
st.sidebar.header("診断設定")
stocks = {
    "テスラ": "TSLA",
    "パランティア": "PLTR",
    "トヨタ": "7203.T",
    "任天堂": "7974.T",
    "エヌビディア": "NVDA",
    "Apple": "AAPL"
}
selected_names = st.sidebar.multiselect("分析する銘柄を選択", list(stocks.keys()), default=["テスラ", "パランティア", "トヨタ"])

# --- 4. 診断ロジック ---
if st.sidebar.button("分析を実行"):
    results = []
    
    with st.spinner('AIが世界情勢と株価を分析中...'):
        for name in selected_names:
            symbol = stocks[name]
            
            # 株価取得 & 予測
            df = yf.download(symbol, period="3mo", progress=False)
            df_study = df.tail(30).copy()
            df_study['Day_Num'] = np.arange(len(df_study))
            
            model = LinearRegression()
            model.fit(df_study[['Day_Num']], df_study['Close'])
            pred_price = model.predict([[len(df_study)]])[0]
            last_price = df['Close'].iloc[-1]
            diff_pct = ((pred_price - last_price) / last_price) * 100
            
            # 日本語ニュース分析
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
                "AI予測(明日)": round(pred_price, 2),
                "期待値(%)": round(diff_pct, 2),
                "ニュース評価": f"{avg_stars:.1f} ★",
                "score": diff_pct + (avg_stars - 3)
            })

    # --- 5. 結果表示 (ランキング) ---
    res_df = pd.DataFrame(results).sort_values(by="score", ascending=False)
    res_df.insert(0, "順位", range(1, len(res_df) + 1))
    
    st.subheader("🏆 注目銘柄ランキング")
    st.table(res_df.drop(columns="score"))

    # --- 6. 視覚化 ---
    st.subheader("📈 トレンド比較")
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in selected_names:
        df = yf.download(stocks[name], period="1mo", progress=False)
        norm_price = df['Close'] / df['Close'].iloc[0] * 100
        ax.plot(df.index, norm_price, label=name)
    
    plt.axhline(100, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    st.pyplot(fig)
    
    st.info("💡 AI予測は統計的なトレンドに基づいています。投資判断は自己責任でお願いします。")
else:
    st.write("サイドバーから銘柄を選んで「分析を実行」を押してください。")
