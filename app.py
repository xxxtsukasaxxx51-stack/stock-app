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
st.set_page_config(page_title="AIマーケット総合診断", layout="wide")

# カスタムCSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .market-box { background-color: #1e1e1e; color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 外部データの取得（為替・市場指数） ---
@st.cache_data(ttl=300) # 5分ごとに更新
def get_market_indices():
    indices = {
        "ドル円": "JPY=X",
        "日経平均": "^N225",
        "NYダウ": "^DJI"
    }
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="2d", progress=False)
            current = info['Close'].iloc[-1]
            prev = info['Close'].iloc[-2]
            diff = current - prev
            data[name] = (current, diff)
        except:
            data[name] = (0, 0)
    return data

indices_data = get_market_indices()

# --- 3. 画面表示 ---
st.title("🌍 AIマーケット総合診断：世界情勢 × 未来予測")

# ★マーケット情報の表示
st.markdown("### 📊 主要マーケット指標")
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric("💴 ドル円", f"{indices_data['ドル円'][0]:.2f}円", f"{indices_data['ドル円'][1]:+.2f}")
with m_col2:
    st.metric("🇯🇵 日経平均", f"{indices_data['日経平均'][0]:,.0f}円", f"{indices_data['日経平均'][1]:+,.0f}")
with m_col3:
    st.metric("🇺🇸 NYダウ", f"{indices_data['NYダウ'][0]:,.0f}ドル", f"{indices_data['NYダウ'][1]:+,.0f}")

st.markdown("---")

# --- 4. AIモデルの読み込み ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

analyzer = load_ai()

# --- 5. サイドバー設定 ---
with st.sidebar:
    st.header("💰 未来シミュレーション")
    future_investment = st.number_input("いま、いくら投資する？(円)", min_value=1000, value=100000, step=10000)
    
    st.header("⚙️ 分析銘柄")
    stocks = {
        "テスラ": "TSLA", "パランティア": "PLTR", "トヨタ": "7203.T",
        "任天堂": "7974.T", "エヌビディア": "NVDA", "Apple": "AAPL",
        "ソニー": "6758.T", "三菱UFJ": "8306.T", "東京エレクトロン": "8035.T"
    }
    selected_names = st.multiselect("銘柄を選択", list(stocks.keys()), default=["エヌビディア", "トヨタ"])
    execute = st.button("🚀 世界情勢と未来を診断")

# --- 6. 実行ロジック ---
if execute:
    results = []
    
    with st.spinner('世界中のニュースと市場データを同期中...'):
        for name in selected_names:
            try:
                symbol = stocks[name]
                df = yf.download(symbol, period="1mo", progress=False)
                current_price = float(df['Close'].iloc[-1])
                
                # AI予測（線形回帰）
                y_data = df['Close'].tail(20).values.reshape(-1, 1)
                X_data = np.arange(len(y_data)).reshape(-1, 1)
                model = LinearRegression(); model.fit(X_data, y_data)
                predicted_price = model.predict([[len(y_data)]])[0][0]
                change_rate = (predicted_price / current_price)
                
                future_value = future_investment * change_rate
                profit_loss = future_value - future_investment
                
                # ニュースと世界情勢の解析
                is_japan = symbol.endswith(".T")
                query = name if is_japan else symbol.split('.')[0]
                lang = "ja" if is_japan else "en"
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl={lang}&gl={'JP' if is_japan else 'US'}"
                feed = feedparser.parse(url)
                
                stars = 3
                topic = "関連ニュースなし"
                if feed.entries:
                    topic = feed.entries[0].title
                    stars = sum([int(analyzer(e.title)[0]['label'].split()[0]) for e in feed.entries[:3]]) / 3
                
                results.append({
                    "銘柄": name,
                    "価格": f"{current_price:,.1f}" + ("円" if is_japan else "ドル"),
                    "明日への予測": f"{future_value:,.0f}円",
                    "損益予想": f"{profit_loss:+,.0f}円",
                    "情勢評価": f"{stars:.1f}★",
                    "注目トピック": topic[:45] + "..."
                })
            except: continue

    if results:
        st.subheader("🏆 個別銘柄の未来診断")
        # リッチな結果表示
        for res in results:
            with st.expander(f"📌 {res['銘柄']} の詳細診断結果", expanded=True):
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.metric("予測資産額", res['予測額' if '予測額' in res else '明日への予測'], res['損益予想'])
                c2.metric("AI情勢スコア", res['情勢評価'])
                c3.write(f"**最新の世界情勢トピック:**\n{res['注目トピック']}")
                
        st.table(pd.DataFrame(results))
    else:
        st.info("左側のメニューから銘柄を選んでボタンを押してください。")

st.caption("※為替・指数・ニュース・統計モデルを組み合わせた総合診断です。最終的な投資判断はご自身の責任で行ってください。")
