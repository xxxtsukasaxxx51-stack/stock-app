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
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 外部データの取得（安全なエラー処理付き） ---
@st.cache_data(ttl=300)
def get_market_indices():
    indices = {
        "ドル円": "JPY=X",
        "日経平均": "^N225",
        "NYダウ": "^DJI"
    }
    data = {}
    for name, ticker in indices.items():
        try:
            # periodを1moにして、直近の有効な2日間を確実に取得
            info = yf.download(ticker, period="1mo", progress=False)
            if len(info) >= 2:
                current = float(info['Close'].iloc[-1])
                prev = float(info['Close'].iloc[-2])
                diff = current - prev
                data[name] = (current, diff)
            else:
                data[name] = (None, None)
        except:
            data[name] = (None, None)
    return data

indices_data = get_market_indices()

# --- 3. 画面表示 ---
st.title("🌍 AIマーケット総合診断：世界情勢 × 未来予測")

st.markdown("### 📊 主要マーケット指標")
m_col1, m_col2, m_col3 = st.columns(3)

# データの有無を確認しながら表示（ここでエラーを防止）
def display_metric(col, label, data_tuple, unit=""):
    val, diff = data_tuple
    if val is not None:
        col.metric(label, f"{val:,.2f}{unit}", f"{diff:+,.2f}")
    else:
        col.metric(label, "取得中...", "市場休止中")

display_metric(m_col1, "💴 ドル円", indices_data['ドル円'], "円")
display_metric(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_metric(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

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
                if len(df) < 5: continue
                
                current_price = float(df['Close'].iloc[-1])
                y_data = df['Close'].tail(20).values.reshape(-1, 1)
                X_data = np.arange(len(y_data)).reshape(-1, 1)
                model = LinearRegression(); model.fit(X_data, y_data)
                predicted_price = float(model.predict([[len(y_data)]])[0][0])
                
                change_rate = predicted_price / current_price
                future_value = future_investment * change_rate
                profit_loss = future_value - future_investment
                
                # ニュース解析
                is_japan = symbol.endswith(".T")
                query = name if is_japan else symbol.split('.')[0]
                lang, gl = ("ja", "JP") if is_japan else ("en", "US")
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl={lang}&gl={gl}"
                feed = feedparser.parse(url)
                
                stars, topic = 3, "関連ニュースなし"
                if feed.entries:
                    topic = feed.entries[0].title
                    stars = sum([int(analyzer(e.title)[0]['label'].split()[0]) for e in feed.entries[:3]]) / 3
                
                results.append({
                    "銘柄": name,
                    "価格": f"{current_price:,.1f}" + ("円" if is_japan else "ドル"),
                    "将来価値": future_value,
                    "損益": profit_loss,
                    "情勢評価": f"{stars:.1f}★",
                    "最新トピック": topic[:45] + "..."
                })
            except: continue

    if results:
        st.subheader("🏆 個別銘柄の未来診断")
        for res in results:
            with st.expander(f"📌 {res['銘柄']} の診断結果", expanded=True):
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.metric("予測資産額", f"{res['将来価値']:,.0f}円", f"{res['損益']:+,.0f}円")
                c2.metric("AI情勢スコア", res['情勢評価'])
                c3.write(f"**最新ニュース:**\n{res['最新トピック']}")
    else:
        st.info("サイドバーから銘柄を選んでボタンを押してください。")

st.caption("※最終的な投資判断はご自身の責任で行ってください。")
