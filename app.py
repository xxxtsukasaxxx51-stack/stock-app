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

# --- 0. グラフ表示の安定化設定 ---
import matplotlib
matplotlib.use('Agg')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AI投資診断(世界対応版)", layout="wide")
st.title("🌍 AI銘柄診断：世界ニュース＆長期トレンド")

# --- 2. AIモデルの読み込み (多言語対応) ---
@st.cache_resource
def load_ai():
    # 英語・日本語を含む多言語を同時に理解できる強力なモデルです
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

analyzer = load_ai()

# --- 3. サイドバー設定 ---
st.sidebar.header("診断設定")
stocks = {
    "テスラ": "TSLA", "パランティア": "PLTR", "トヨタ": "7203.T",
    "任天堂": "7974.T", "エヌビディア": "NVDA", "Apple": "AAPL",
    "ソニー": "6758.T", "ソフトバンクG": "9984.T"
}
selected_names = st.sidebar.multiselect("分析する銘柄", list(stocks.keys()), default=["テスラ", "エヌビディア", "トヨタ"])
time_span = st.sidebar.radio("表示スパン", ["1年", "5年", "10年"])
span_map = {"1年": "1y", "5年": "5y", "10年": "10y"}

# --- 4. 実行ボタン ---
if st.sidebar.button("分析を実行"):
    results = []
    plot_data = {} 
    
    with st.spinner('世界中のニュースと株価を収集中...'):
        for name in selected_names:
            try:
                symbol = stocks[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if len(df) < 20: continue
                plot_data[name] = df

                # AI予測計算
                y = df['Close'].tail(30).values.reshape(-1, 1)
                X = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression(); model.fit(X, y)
                pred_price = model.predict([[len(y)]])[0][0]
                last_price = float(y[-1][0])
                diff_pct = ((pred_price - last_price) / last_price) * 100
                
                # --- 【新機能】日本と世界のニュース使い分け ---
                is_japan = symbol.endswith(".T")
                if is_japan:
                    # 日本株：日本のGoogleニュース(日本語)
                    query = urllib.parse.quote(name)
                    url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
                else:
                    # 米国株：米国のGoogleニュース(英語)
                    query = urllib.parse.quote(symbol.split('.')[0])
                    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
                
                feed = feedparser.parse(url)
                stars, count = 0, 0
                top_news = "ニュースが見つかりませんでした"
                
                if feed.entries:
                    top_news = feed.entries[0].title # 最新の1件を保持
                    for entry in feed.entries[:3]: # 直近3件を分析
                        res = analyzer(entry.title)[0]
                        stars += int(res['label'].split()[0])
                        count += 1
                avg_stars = stars / count if count > 0 else 3
                
                results.append({
                    "銘柄": name, "現在価格": round(last_price, 2),
                    "AI予測(明日)": round(float(pred_price), 2),
                    "期待値(%)": round(float(diff_pct), 2),
                    "ニュース評価": f"{avg_stars:.1f} ★",
                    "最新ニュース": top_news[:50] + "...", # タイトルを表示
                    "score": float(diff_pct) + (avg_stars - 3)
                })
            except: continue

    if results:
        # ランキング表示
        res_df = pd.DataFrame(results).sort_values(by="score", ascending=False)
        st.subheader(f"🏆 AI総合評価ランキング")
        # ニュースタイトルを含めて表示
        st.dataframe(res_df.drop(columns="score"), use_container_width=True)

        # グラフ表示
        st.subheader(f"📈 {time_span}トレンド ＆ 明日の予測(★)")
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, data in plot_data.items():
            norm_price = data['Close'] / data['Close'].iloc[0] * 100
            line = ax.plot(data.index, norm_price, label=name, alpha=0.8)
            
            # 予測点(★)の描画
            next_date = data.index[-1] + pd.Timedelta(days=1)
            pred_val = [r['AI予測(明日)'] for r in results if r['銘柄']==name][0]
            norm_pred = (pred_val / data['Close'].iloc[0]) * 100
            ax.scatter(next_date, norm_pred, color=line[0].get_color(), marker='*', s=250, edgecolors='black', zorder=5)
        
        plt.axhline(100, color='black', linestyle='--', alpha=0.3)
        plt.legend()
        st.pyplot(fig)
    else:
        st.error("データ取得に失敗しました。")
