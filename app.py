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
from googletrans import Translator # 和訳用ライブラリ

# --- 0. グラフ表示の安定化設定 ---
import matplotlib
matplotlib.use('Agg')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide")

# カスタムCSS
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .news-box { background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #007bff; margin-bottom: 8px; }
    .news-title-jp { font-weight: bold; color: #333; margin-bottom: 4px; }
    .news-title-en { font-size: 0.8em; color: #888; font-style: italic; }
    .advice-box { padding: 15px; border-radius: 10px; margin-top: 10px; font-weight: bold; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 翻訳・AIモデルの準備 ---
@st.cache_resource
def load_models():
    analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    translator = Translator()
    return analyzer, translator

analyzer, translator = load_models()

# --- 3. 指標データの取得 ---
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

# --- 4. 画面表示 ---
st.title("🌍 AIマーケット総合診断 Pro (和訳対応)")

m_col1, m_col2, m_col3 = st.columns(3)
def display_metric(col, label, data_tuple, unit=""):
    val, diff = data_tuple
    if val is not None: col.metric(label, f"{val:,.2f}{unit}", f"{diff:+,.2f}")
    else: col.metric(label, "取得中...", "市場休止中")

display_metric(m_col1, "💴 ドル円", indices_data['ドル円'], "円")
display_metric(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_metric(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

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
    st.subheader("✍️ 自由に入力")
    custom_symbol = st.text_input("例: NFLX, 6752.T", "")
    if custom_symbol:
        custom_name = f"自由入力({custom_symbol})"
        all_stocks[custom_name] = custom_symbol
        if custom_name not in selected_names: selected_names.append(custom_name)
    
    st.markdown("---")
    future_investment = st.number_input("投資金額(円)", min_value=1000, value=100000)
    time_span = st.select_slider("期間", options=["1週間", "30日", "1年", "5年"], value="30日")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y"}
    execute = st.button("🚀 総合診断を実行")

# --- 6. 実行ロジック ---
if execute:
    results = []
    plot_data = {}
    
    with st.spinner('世界中のニュースを和訳・分析中...'):
        for name in selected_names:
            try:
                symbol = all_stocks[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data[name] = df
                
                # 未来予測
                current_price = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model = LinearRegression().fit(X_reg, y_reg)
                pred_p = float(model.predict([[len(y_reg)]])[0][0])
                
                # ニュース取得
                is_j = ".T" in symbol
                search_q = name.split("(")[-1].replace(")", "") if "自由入力" in name else (name if is_j else symbol)
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url)
                
                news_details, stars_sum = [], 0
                if feed.entries:
                    for entry in feed.entries[:3]:
                        # 感情分析
                        score = int(analyzer(entry.title)[0]['label'].split()[0])
                        stars_sum += score
                        # ★和訳処理
                        title_jp = entry.title
                        if not is_j: # 米国銘柄なら和訳
                            try:
                                title_jp = translator.translate(entry.title, dest='ja').text
                            except: pass
                        news_details.append({"title_jp": title_jp, "title_en": entry.title, "score": score})
                    avg_stars = stars_sum / len(news_details)
                else: avg_stars = 3
                
                # アドバイスロジック
                trend_up = pred_p > current_price
                if avg_stars >= 3.5 and trend_up: advice, color = "🌟【絶好調】", "#e8f5e9"
                elif avg_stars <= 2.5 and not trend_up: advice, color = "⚠️【警戒】", "#ffebee"
                elif avg_stars <= 2.5 and trend_up: advice, color = "🤔【チグハグ】悪いニュースに耐えています。", "#fff3e0"
                elif avg_stars >= 3.5 and not trend_up: advice, color = "❓【チグハグ】いいニュースなのに元気なし。", "#e1f5fe"
                else: advice, color = "😐【様子見】", "#f5f5f5"

                results.append({
                    "銘柄": name, "将来価値": future_investment * (pred_p / current_price), 
                    "評価": avg_stars, "pred": pred_p, "news": news_details,
                    "symbol": symbol, "advice": advice, "color": color
                })
            except: continue

    if results:
        # グラフ表示
        st.subheader("📈 トレンド予測グラフ")
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, data in plot_data.items():
            base_p = data['Close'].iloc[0]
            norm_p = data['Close'] / base_p * 100
            line = ax.plot(data.index, norm_p, label=name, linewidth=2)
            res_item = next(r for r in results if r['銘柄'] == name)
            norm_pred = (res_item['pred'] / base_p) * 100
            future_date = data.index[-1] + timedelta(days=1)
            ax.plot([data.index[-1], future_date], [norm_p.iloc[-1], norm_pred], color=line[0].get_color(), linestyle='--', alpha=0.5)
            ax.scatter(future_date, norm_pred, color=line[0].get_color(), marker='*', s=350, edgecolors='black', zorder=10)
        st.pyplot(fig)

        # 診断詳細
        st.markdown("---")
        st.subheader("🏆 AI診断詳細 & 和訳ニュース")
        for res in results:
            with st.expander(f"📌 {res['銘柄']} の診断", expanded=True):
                col_m, col_n = st.columns([1, 2])
                with col_m:
                    st.metric("明日への予測額", f"{res['将来価値']:,.0f}円", f"{res['将来価値']-future_investment:+,.0f}円")
                    st.write(f"**AI評価:** {res['評価']:.1f} ★")
                    st.markdown(f"<div class='advice-box' style='background-color: {res['color']};'>{res['advice']}</div>", unsafe_allow_html=True)
                with col_n:
                    st.write("**世界の最新ヘッドライン (AI和訳):**")
                    for n in res['news']:
                        st.markdown(f"""<div class='news-box'>{'⭐' * n['score']}<br>
                        <div class='news-title-jp'>{n['title_jp']}</div>
                        <div class='news-title-en'>{n['title_en']}</div></div>""", unsafe_allow_html=True)
    else: st.error("分析に失敗しました。")

# --- 最後に重要：ライブラリのインストール指示 ---
# GitHubの requirements.txt に googletrans==4.0.0-rc1 を追加する必要があります。
