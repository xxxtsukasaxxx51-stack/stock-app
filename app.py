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
st.set_page_config(page_title="AI投資シミュレーター", layout="wide")

# カスタムCSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("💰 AI投資診断 & 損益シミュレーター")
st.markdown("---")

# --- 2. AIモデルの読み込み ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

analyzer = load_ai()

# --- 3. サイドバー設定 ---
with st.sidebar:
    st.header("⚙️ シミュレーション設定")
    # ★追加：投資金額の設定
    investment_amount = st.number_input("もし、開始日にいくら投資してたら？(円)", min_value=1000, value=100000, step=10000)
    
    stocks = {
        "テスラ": "TSLA", "パランティア": "PLTR", "トヨタ": "7203.T",
        "任天堂": "7974.T", "エヌビディア": "NVDA", "Apple": "AAPL",
        "ソニー": "6758.T", "ソフトバンクG": "9984.T"
    }
    selected_names = st.multiselect("分析銘柄", list(stocks.keys()), default=["エヌビディア", "テスラ", "トヨタ"])
    time_span = st.select_slider("シミュレーション期間", options=["1週間", "30日", "1年", "5年", "10年"], value="1年")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y", "10年": "10y"}
    
    execute = st.button("🚀 シミュレーション実行")

# --- 4. 実行ロジック ---
if execute:
    results = []
    plot_data = {} 
    
    with st.spinner('過去のデータとAI予測を計算中...'):
        for name in selected_names:
            try:
                symbol = stocks[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if len(df) < 2: continue
                plot_data[name] = df

                # 損益計算
                start_price = float(df['Close'].iloc[0])
                current_price = float(df['Close'].iloc[-1])
                return_rate = (current_price / start_price)
                
                # 今の価値 = 投資額 × 騰落率
                current_value = investment_amount * return_rate
                profit_loss = current_value - investment_amount

                # AI予測（明日）
                y_data = df['Close'].tail(30).values.reshape(-1, 1)
                X_data = np.arange(len(y_data)).reshape(-1, 1)
                model = LinearRegression(); model.fit(X_data, y_data)
                pred_price = model.predict([[len(y_data)]])[0][0]
                diff_pct = ((pred_price - current_price) / current_price) * 100
                
                # ニュース評価
                is_japan = symbol.endswith(".T")
                lang_url = f"&hl=ja&gl=JP&ceid=JP:ja" if is_japan else f"&hl=en-US&gl=US&ceid=US:en"
                query = name if is_japan else symbol.split('.')[0]
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}{lang_url}"
                feed = feedparser.parse(url)
                
                stars = sum([int(analyzer(e.title)[0]['label'].split()[0]) for e in feed.entries[:3]]) / 3 if feed.entries else 3
                
                results.append({
                    "銘柄": name,
                    "開始時価格": f"${start_price:.2f}" if not is_japan else f"{start_price:.0f}円",
                    "現在価格": f"${current_price:.2f}" if not is_japan else f"{current_price:.0f}円",
                    "今の価値": f"{current_value:,.0f}円",
                    "損益": f"{profit_loss:+,.0f}円",
                    "AI評価": f"{stars:.1f}★",
                    "明日予測": f"{diff_pct:+.2f}%",
                    "raw_diff": diff_pct,
                    "raw_stars": stars
                })
            except: continue

    if results:
        # --- レイアウト: シミュレーション結果 ---
        st.subheader(f"📊 {time_span}前に {investment_amount:,.0f}円 投資していたら？")
        
        # 損益をカード形式で並べる
        cols = st.columns(len(results))
        for i, res in enumerate(results):
            with cols[i]:
                st.metric(label=res['銘柄'], value=res['今の価値'], delta=res['損益'])

        st.markdown("---")
        
        # --- レイアウト: 詳細ランキング ---
        col_t, col_g = st.columns([1.2, 1])
        with col_t:
            st.subheader("🏆 AI総合診断ランキング")
            res_df = pd.DataFrame(results).sort_values(by="raw_stars", ascending=False)
            st.table(res_df[["銘柄", "開始時価格", "現在価格", "AI評価", "明日予測"]])
            
        with col_g:
            st.subheader("📈 成長率の比較 (%)")
            fig, ax = plt.subplots(figsize=(10, 7))
            for name, data in plot_data.items():
                norm_price = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                ax.plot(data.index, norm_price, label=name, linewidth=2)
            
            plt.axhline(0, color='black', linestyle='--', alpha=0.3)
            plt.ylabel("損益率 (%)")
            plt.legend()
            st.pyplot(fig)
    else:
        st.info("サイドバーから銘柄を選んで『実行』を押してください。")

st.info("※日本株は円、米国株はドルベースの騰落をベースに簡易計算しています。")
