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
from datetime import timedelta

# --- 0. グラフ表示の安定化設定 ---
import matplotlib
matplotlib.use('Agg')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide")

# --- 2. 指標データの取得 ---
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

# --- 3. メイン画面 ---
st.title("🌍 AIマーケット総合診断 Pro")

m_col1, m_col2, m_col3 = st.columns(3)
def display_metric(col, label, data_tuple, unit=""):
    val, diff = data_tuple
    if val is not None: col.metric(label, f"{val:,.2f}{unit}", f"{diff:+,.2f}")
    else: col.metric(label, "取得中...", "市場休止中")

display_metric(m_col1, "💴 ドル円", indices_data['ドル円'], "円")
display_metric(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_metric(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

# --- 4. AIモデル読み込み ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
analyzer = load_ai()

# --- 5. サイドバー ---
with st.sidebar:
    st.header("🔍 銘柄の選択")
    stock_presets = {
        "🇺🇸 米国人気株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "パランティア": "PLTR"},
        "🇯🇵 日本人気株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T"},
        "⚡ 暗号資産/他": {"ビットコイン": "BTC-USD", "金(Gold)": "GC=F"}
    }
    all_stocks = {}
    for cat, items in stock_presets.items(): all_stocks.update(items)
    selected_names = st.multiselect("リストから選択", list(all_stocks.keys()), default=["エヌビディア", "トヨタ"])
    
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
                
                # ニュース解析
                is_j = ".T" in symbol
                q = name if is_j else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url)
                stars = sum([int(analyzer(e.title)[0]['label'].split()[0]) for e in feed.entries[:3]]) / 3 if feed.entries else 3
                
                results.append({"銘柄": name, "将来価値": future_investment * (pred_p / current_price), "評価": stars, "pred": pred_p, "current": current_price})
            except: continue

    if results:
        st.subheader("📈 トレンド予測グラフ")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, data in plot_data.items():
            # 1. 過去のデータをプロット（開始を100とする）
            base_price = data['Close'].iloc[0]
            norm_p = data['Close'] / base_price * 100
            line = ax.plot(data.index, norm_p, label=name, linewidth=2, marker='o' if time_span=="1週間" else None)
            color = line[0].get_color()
            
            # 2. 未来の日付を計算（最新の日の翌日）
            last_date = data.index[-1]
            future_date = last_date + timedelta(days=1)
            
            # 3. 予測値を正規化してプロット
            res_item = next(r for r in results if r['銘柄'] == name)
            norm_pred = (res_item['pred'] / base_price) * 100
            
            # 4. 最新点と予測星印を点線で結ぶ
            ax.plot([last_date, future_date], [norm_p.iloc[-1], norm_pred], color=color, linestyle='--', alpha=0.6)
            
            # 5. 未来の地点に大きな星を描画
            ax.scatter(future_date, norm_pred, color=color, marker='*', s=400, edgecolors='black', zorder=10, label=f"{name} 予測")
        
        plt.axhline(100, color='black', linestyle='-', alpha=0.1)
        plt.title(f"株価推移とAIによる明日予測 ({time_span})", fontsize=14)
        plt.ylabel("成長率 (%)")
        plt.grid(True, alpha=0.2)
        # 凡例を整理（予測と実線を分ける）
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(selected_names)], labels[:len(selected_names)], loc='upper left')
        st.pyplot(fig)

        # 診断結果を下に表示
        st.markdown("---")
        st.subheader("🏆 AI診断詳細")
        cols = st.columns(len(results))
        for i, res in enumerate(results):
            with cols[i]:
                st.metric(res['銘柄'], f"{res['将来価値']:,.0f}円", f"{res['将来価値']-future_investment:+,.0f}円")
                st.write(f"AI情勢評価: {res['評価']:.1f} ★")
    else:
        st.error("データが取得できませんでした。")

st.info("💡 グラフの点線と★は、過去のトレンドからAIが導き出した『明日の着地予想』です。")
