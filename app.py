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
st.set_page_config(page_title="AI投資診断(初心者ガイド付)", layout="wide")
st.title("🌍 AI銘柄診断：世界ニュース＆長期トレンド")

# --- 2. AIモデルの読み込み ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

analyzer = load_ai()

# --- 3. サイドバー設定 ---
st.sidebar.header("診断設定")
stocks = {
    "テスラ": "TSLA", "パランティア": "PLTR", "トヨタ": "7203.T",
    "任天堂": "7974.T", "エヌビディア": "NVDA", "Apple": "AAPL",
    "ソニー": "6758.T", "ソフトバンクG": "9984.T"
}
selected_names = st.sidebar.multiselect("分析する銘柄を選択", list(stocks.keys()), default=["テスラ", "エヌビディア", "トヨタ"])
time_span = st.sidebar.radio("表示スパン（期間）", ["1週間", "30日", "1年", "5年", "10年"], index=1)
span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y", "10年": "10y"}

# --- 初心者向け：ニュース評価の解説パネル ---
with st.expander("💡 ニュース評価の仕組み（初めての方へ）"):
    st.write("""
    このアプリのAIは、世界中のニュース見出しを読んで、その内容が**「お祝いムード（株が上がりそう）」**か**「悲観ムード（下がりそう）」**かを判定しています。
    * **★5.0 (絶好調)**：明るいニュースが多く、期待が高まっています。
    * **★3.0 (普通)**：特に大きなニュースがないか、良い悪いが半々の状態です。
    * **★1.0 (注意)**：トラブルや業績不振などのニュースが目立っています。
    """)

# --- 4. 実行ボタン ---
if st.sidebar.button("分析を実行"):
    results = []
    plot_data = {} 
    
    with st.spinner('AIが最新情報を分析中...'):
        for name in selected_names:
            try:
                symbol = stocks[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if len(df) < 2: continue
                plot_data[name] = df

                # 予測計算
                y_data = df['Close'].tail(30).values.reshape(-1, 1)
                X_data = np.arange(len(y_data)).reshape(-1, 1)
                model = LinearRegression(); model.fit(X_data, y_data)
                pred_price = model.predict([[len(y_data)]])[0][0]
                last_price = float(df['Close'].iloc[-1])
                diff_pct = ((pred_price - last_price) / last_price) * 100
                
                # ニュース取得
                is_japan = symbol.endswith(".T")
                if is_japan:
                    query = urllib.parse.quote(name)
                    url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
                else:
                    query = urllib.parse.quote(symbol.split('.')[0])
                    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
                
                feed = feedparser.parse(url)
                stars, count, top_news = 0, 0, "なし"
                if feed.entries:
                    top_news = feed.entries[0].title
                    for entry in feed.entries[:3]:
                        res = analyzer(entry.title)[0]
                        stars += int(res['label'].split()[0])
                        count += 1
                avg_stars = stars / count if count > 0 else 3
                
                status = "😊 期待" if avg_stars > 3.5 else "😐 中立" if avg_stars >= 2.5 else "⚠️ 注意"
                
                results.append({
                    "銘柄": name, "現在価格": round(last_price, 2),
                    "AI予測(明日)": round(float(pred_price), 2),
                    "AI判定": status,
                    "評価詳細": f"{avg_stars:.1f} ★",
                    "最新ニュース": top_news[:40] + "...",
                    "score": float(diff_pct) + (avg_stars - 3)
                })
            except: continue

    if results:
        # --- ここが修正ポイント！ ---
        res_df = pd.DataFrame(results).sort_values(by="score", ascending=False)
        st.subheader(f"🏆 AI診断ランキング")
        st.dataframe(res_df.drop(columns="score"), use_container_width=True)

        st.subheader(f"📈 {time_span}の推移 ＆ 明日予測(★)")
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, data in plot_data.items():
            norm_price = data['Close'] / data['Close'].iloc[0] * 100
            m = 'o' if time_span in ["1週間", "30日"] else None
            line = ax.plot(data.index, norm_price, label=name, alpha=0.8, marker=m)
            
            # 予測点の描画
            next_date = data.index[-1] + pd.Timedelta(days=1)
            # 各銘柄の予測値を探す
            pred_val = [r['AI予測(明日)'] for r in results if r['銘柄']==name][0]
            norm_pred = (pred_val / data['Close'].iloc[0]) * 100
            ax.scatter(next_date, norm_pred, color=line[0].get_color(), marker='*', s=300, edgecolors='black', zorder=5)
        
        plt.axhline(100, color='black', linestyle='--', alpha=0.3)
        plt.legend()
        st.pyplot(fig)
    else:
        st.error("分析に失敗しました。銘柄を選び直してください。")

st.info("💡 グラフの星印(★)は明日への期待値です。")
