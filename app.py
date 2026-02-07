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

# --- 1. ページ設定 (テーマカラーを意識) ---
st.set_page_config(page_title="AI投資診断 Premium", layout="wide", initial_sidebar_state="expanded")

# --- カスタムCSSでデザインを整える ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; border: none; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("💎 AI銘柄診断 Premium")
st.markdown("---")

# --- 2. AIモデルの読み込み ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

analyzer = load_ai()

# --- 3. サイドバー設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    stocks = {
        "テスラ": "TSLA", "パランティア": "PLTR", "トヨタ": "7203.T",
        "任天堂": "7974.T", "エヌビディア": "NVDA", "Apple": "AAPL",
        "ソニー": "6758.T", "ソフトバンクG": "9984.T"
    }
    selected_names = st.multiselect("分析銘柄を選択", list(stocks.keys()), default=["エヌビディア", "テスラ"])
    time_span = st.select_slider("表示期間", options=["1週間", "30日", "1年", "5年", "10年"], value="30日")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y", "10年": "10y"}
    
    st.markdown("---")
    execute = st.button("🚀 分析を開始する")

# --- 解説パネル ---
with st.expander("❓ ニュース評価とは？"):
    st.info("世界中の最新ニュースをAIが読み取り、投資家の感情を1.0〜5.0の星数で数値化しています。")

# --- 4. 実行ロジック ---
if execute:
    results = []
    plot_data = {} 
    
    with st.spinner('✨ AIが市場の波動を解析中...'):
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
                    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(name)}&hl=ja&gl=JP&ceid=JP:ja"
                else:
                    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(symbol.split('.')[0])}&hl=en-US&gl=US&ceid=US:en"
                
                feed = feedparser.parse(url)
                stars, count, news_title = 0, 0, "ニュースなし"
                if feed.entries:
                    news_title = feed.entries[0].title
                    for entry in feed.entries[:3]:
                        res = analyzer(entry.title)[0]
                        stars += int(res['label'].split()[0])
                        count += 1
                avg_stars = stars / count if count > 0 else 3
                
                results.append({
                    "name": name, "price": last_price, "pred": pred_price, 
                    "diff": diff_pct, "stars": avg_stars, "news": news_title
                })
            except: continue

    if results:
        # --- レイアウト1: メトリクス表示 ---
        st.subheader("📊 リアルタイム要約")
        cols = st.columns(len(results))
        for i, res in enumerate(results):
            with cols[i]:
                color = "normal" if res['diff'] >= 0 else "inverse"
                st.metric(label=res['name'], value=f"${res['price']:.2f}", delta=f"{res['diff']:.2f}% (明日予測)", delta_color=color)

        # --- レイアウト2: ランキングとグラフ ---
        col_table, col_graph = st.columns([1, 1.5])
        
        with col_table:
            st.subheader("🏆 総合評価")
            res_df = pd.DataFrame(results).sort_values(by="stars", ascending=False)
            st.table(res_df[["name", "stars", "news"]].rename(columns={"name":"銘柄", "stars":"AI評価", "news":"最新ニュース"}))

        with col_graph:
            st.subheader("📈 トレンド予測")
            plt.style.use('ggplot') # おしゃれなグラフスタイル
            fig, ax = plt.subplots(figsize=(10, 6))
            for name, data in plot_data.items():
                norm_price = data['Close'] / data['Close'].iloc[0] * 100
                line = ax.plot(data.index, norm_price, label=name, linewidth=2)
                
                # 予測地点に星
                pred_val = [r['pred'] for r in results if r['name']==name][0]
                norm_pred = (pred_val / data['Close'].iloc[0]) * 100
                ax.scatter(data.index[-1] + pd.Timedelta(days=1), norm_pred, color=line[0].get_color(), marker='*', s=300, edgecolors='black', zorder=5)
            
            plt.axhline(100, color='#333333', linestyle='--', alpha=0.2)
            plt.legend()
            st.pyplot(fig)
    else:
        st.error("分析対象を選択して実行してください。")
