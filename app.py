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
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide")

# カスタムCSS
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stMultiSelect div[data-baseweb="select"] { background-color: #e3f2fd; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 指標データの取得（安全版） ---
@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"ドル円": "JPY=X", "日経平均": "^N225", "NYダウ": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="1mo", progress=False)
            if not info.empty:
                current = float(info['Close'].iloc[-1])
                prev = float(info['Close'].iloc[-2])
                data[name] = (current, current - prev)
            else: data[name] = (None, None)
        except: data[name] = (None, None)
    return data

indices_data = get_market_indices()

# --- 3. メイン画面表示 ---
st.title("🌍 AIマーケット総合診断 Pro")

m_col1, m_col2, m_col3 = st.columns(3)
def display_metric(col, label, data_tuple, unit=""):
    val, diff = data_tuple
    if val is not None: col.metric(label, f"{val:,.2f}{unit}", f"{diff:+,.2f}")
    else: col.metric(label, "取得中...", "市場休止中")

display_metric(m_col1, "💴 ドル円", indices_data['ドル円'], "円")
display_metric(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_metric(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

st.markdown("---")

# --- 4. AIモデル読み込み ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
analyzer = load_ai()

# --- 5. サイドバー設定（銘柄リストの大幅拡充） ---
with st.sidebar:
    st.header("🔍 銘柄の選択")
    
    # カテゴリー別銘柄リスト
    stock_presets = {
        "🇺🇸 米国人気株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "Amazon": "AMZN", "Microsoft": "MSFT", "Google": "GOOGL", "Meta": "META", "パランティア": "PLTR"},
        "🇯🇵 日本人気株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T", "ソフトバンクG": "9984.T", "キーエンス": "6861.T", "ファーストリテイリング": "9983.T"},
        "⚡ 暗号資産/他": {"ビットコイン": "BTC-USD", "イーサリアム": "ETH-USD", "金(Gold)": "GC=F"}
    }
    
    # プリセットから選ぶ
    all_stocks = {}
    for cat, items in stock_presets.items():
        all_stocks.update(items)
        
    selected_names = st.multiselect("リストから選択", list(all_stocks.keys()), default=["エヌビディア", "トヨタ"])
    
    # ★自由入力機能を追加
    st.markdown("---")
    st.subheader("✍️ 自由に入力 (Yahoo Finance Symbol)")
    custom_symbol = st.text_input("例: NFLX (Netflix), 6752.T (パナソニック)", "")
    if custom_symbol:
        symbol_name = f"カスタム({custom_symbol})"
        all_stocks[symbol_name] = custom_symbol
        if symbol_name not in selected_names:
            selected_names.append(symbol_name)

    st.markdown("---")
    future_investment = st.number_input("投資金額(円)", min_value=1000, value=100000)
    time_span = st.select_slider("期間", options=["1週間", "30日", "1年", "5年"], value="30日")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y"}
    execute = st.button("🚀 総合診断を実行")

# --- 6. 実行ロジック ---
if execute:
    results = []
    plot_data = {}
    
    with st.spinner('データを解析中...'):
        for name in selected_names:
            try:
                symbol = all_stocks[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data[name] = df
                
                # 未来予測計算
                current_price = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model = LinearRegression().fit(X_reg, y_reg)
                pred_p = float(model.predict([[len(y_reg)]])[0][0])
                future_val = future_investment * (pred_p / current_price)
                
                # ニュース解析
                is_j = ".T" in symbol
                q = name.replace("カスタム(", "").replace(")", "") if "カスタム" in name else (name if is_j else symbol)
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url)
                stars = sum([int(analyzer(e.title)[0]['label'].split()[0]) for e in feed.entries[:3]]) / 3 if feed.entries else 3
                
                results.append({"銘柄": name, "将来価値": future_val, "評価": stars, "pred": pred_p, "current": current_price})
            except: continue

    if results:
        st.subheader("🏆 未来診断結果")
        for res in results:
            with st.expander(f"📌 {res['銘柄']} の診断", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("明日への予測額", f"{res['将来価値']:,.0f}円", f"{res['将来価値']-future_investment:+,.0f}円")
                c2.metric("AI情勢評価", f"{res['評価']:.1f} ★")

        st.subheader("📈 トレンド比較")
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, data in plot_data.items():
            norm_p = data['Close'] / data['Close'].iloc[0] * 100
            line = ax.plot(data.index, norm_p, label=name, linewidth=2)
            res_item = next(r for r in results if r['銘柄'] == name)
            norm_pred = (res_item['pred'] / data['Close'].iloc[0]) * 100
            ax.scatter(data.index[-1] + pd.Timedelta(days=1), norm_pred, color=line[0].get_color(), marker='*', s=300, edgecolors='black', zorder=5)
        
        plt.axhline(100, color='black', linestyle='--', alpha=0.2)
        plt.legend()
        st.pyplot(fig)
    else:
        st.error("データが取得できませんでした。シンボルが正しいか確認してください。")
